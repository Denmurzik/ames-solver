import os
import io
import json
import time
import hashlib
import logging
import zipfile
import redis
from flask import Flask, request, jsonify, render_template, Response, stream_with_context, send_file
from celery import Celery
from db import init_db, save_test, get_all_tests

# Logging
logger = logging.getLogger('stealth')
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(handler)


def _order_independent_hash(text):
    """Hash that is the same regardless of line order (handles shuffled options)."""
    raw_lines = str(text).lower().strip().split('\n')
    cleaned = []
    for line in raw_lines:
        l = line.strip()
        if not l:
            continue
        # Simple stripping: remove common option prefixes
        for prefix_len in [4, 3, 2]:  # "15) ", "a) ", "1) ", "a.", "1."
            if len(l) > prefix_len:
                head = l[:prefix_len]
                if (len(head) >= 2 and head[-1] in ').' and head[-2].isalnum()):
                    l = l[prefix_len:].lstrip()
                    break
        # Remove bullet markers
        if l and l[0] in '-*':
            l = l[1:].lstrip()
        l = ' '.join(l.split())
        if l:
            cleaned.append(l)
    cleaned.sort()
    norm = '\n'.join(cleaned)
    return hashlib.md5(norm.encode('utf-8')).hexdigest()

app = Flask(__name__)
init_db()

# Lightweight Celery Client (no tasks/ML models imported)
celery_client = Celery('stealth_tasks', broker='redis://127.0.0.1:6379/0', backend='redis://127.0.0.1:6379/1')

# Redis Client for caching
redis_client = redis.Redis(host='127.0.0.1', port=6379, db=2, decode_responses=True)

# Redis Client for Pub/Sub (separate connection, no decode_responses for pubsub)
redis_pubsub_pool = redis.Redis(host='127.0.0.1', port=6379, db=2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/solve_stream", methods=["POST"])
def solve_stream():
    data = request.json
    test_text = data.get("text", "")

    if not test_text:
        return jsonify({"success": False, "error": "No text provided"}), 400

    def generate():
        # Track tasks we OWN (we dispatched the Celery task and hold the Redis lock)
        owned_tasks = {}   # {i: {"task_id": str, "q_text": str, "hash": str}}
        # Track tasks we SUBSCRIBE to (another request is solving, we just wait for the answer)
        subscribed_tasks = {}  # {i: {"q_text": str, "hash": str}}

        try:
            # 1. Выделяем вопросы через Celery (с дедупликацией + retry)
            yield f"data: {json.dumps({'type': 'status', 'message': 'Parsing questions...'})}\n\n"
            
            t_start = time.time()
            logger.info(f"")
            logger.info(f"==================================================")
            logger.info(f">> NEW TEST FROM USER ({len(test_text)} chars)")

            # Persist raw test text to SQLite
            try:
                save_test(test_text)
                logger.info(f"[DB] Test saved ({len(test_text)} chars)")
            except Exception as db_err:
                logger.error(f"[DB] Failed to save test: {db_err}")
            logger.info(f"==================================================")

            # Dedup extraction by test text hash
            norm_test = ' '.join(test_text.lower().split())
            test_hash = hashlib.md5(norm_test.encode('utf-8')).hexdigest()
            ext_cache_key = f"stealth:ext:{test_hash}"
            ext_lock_key = f"stealth:ext_pending:{test_hash}"
            questions = None

            # Check extraction cache
            cached_ext = redis_client.get(ext_cache_key)
            if cached_ext:
                try:
                    questions = json.loads(cached_ext)
                    logger.info(f"[CACHE] Extraction from cache ({len(questions)} questions)")
                except:
                    pass

            if questions is None:
                lock_acquired = redis_client.set(ext_lock_key, "1", nx=True, ex=300)
                if lock_acquired:
                    # Owner: extract questions
                    logger.info(f"[Step 1] Extracting questions via Gemini... (owner)")
                    for attempt in range(2):
                        try:
                            ext_task = celery_client.send_task('stealth_tasks.extract_questions', args=[test_text])
                            questions = celery_client.AsyncResult(ext_task.id).get(timeout=120)
                            if questions and len(questions) > 0:
                                redis_client.set(ext_cache_key, json.dumps(questions), ex=600)
                                redis_client.delete(ext_lock_key)
                                redis_pubsub_pool.publish(f"stealth:ext_done:{test_hash}", "1")
                                break
                            else:
                                logger.warning(f"[!] Extraction returned 0 questions (attempt {attempt+1}/2)")
                                questions = None
                        except Exception as e:
                            logger.error(f"[FAIL] Extraction error (attempt {attempt+1}/2): {e}")
                            questions = None
                    if questions is None:
                        redis_client.delete(ext_lock_key)
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Could not extract questions after retries.'})}\n\n"
                        return
                else:
                    # Subscriber: wait for owner's extraction result
                    logger.info(f"[Step 1] Waiting for extraction from another user... (subscriber)")
                    ext_pubsub = redis_pubsub_pool.pubsub()
                    ext_pubsub.subscribe(f"stealth:ext_done:{test_hash}")
                    wait_start = time.time()
                    while time.time() - wait_start < 150:
                        ext_pubsub.get_message(timeout=1.0)
                        cached_ext = redis_client.get(ext_cache_key)
                        if cached_ext:
                            try:
                                questions = json.loads(cached_ext)
                                logger.info(f"[CACHE] Got extraction from owner via pubsub ({len(questions)} questions)")
                                break
                            except:
                                pass
                        if not redis_client.exists(ext_lock_key):
                            # Owner disappeared - takeover
                            logger.warning(f"[!] Extraction owner disappeared, taking over...")
                            try:
                                ext_task = celery_client.send_task('stealth_tasks.extract_questions', args=[test_text])
                                questions = celery_client.AsyncResult(ext_task.id).get(timeout=120)
                                if questions and len(questions) > 0:
                                    redis_client.set(ext_cache_key, json.dumps(questions), ex=600)
                            except Exception as e:
                                logger.error(f"[FAIL] Takeover extraction failed: {e}")
                            break
                    ext_pubsub.unsubscribe()
                    ext_pubsub.close()
                    if questions is None or len(questions) == 0:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Could not extract questions from text.'})}\n\n"
                        return

            total = len(questions)
            extract_time = time.time() - t_start
            logger.info(f"[OK] Extracted {total} questions in {extract_time:.1f}s")


            yield f"data: {json.dumps({'type': 'init', 'total': total})}\n\n"

            base_prompt = "Ты — ИИ-ассистент, реши тест."
            try:
                with open("/home/denis/ames-suck-my-ass/prompt.txt", "r", encoding="utf-8") as f:
                    p_content = f.read().strip()
                    if p_content: base_prompt = p_content
            except: pass

            results_output = [None] * total
            completed = 0

            # 2. Проверяем кэш, пытаемся захватить блокировку, или подписываемся
            logger.info(f"")
            logger.info(f"[Step 2] Distributing {total} questions (cache / dispatch / subscribe)")
            logger.info(f"{'~'*50}")
            for i, q_text in enumerate(questions):
                q_hash = _order_independent_hash(q_text)
                q_short = str(q_text)[:50].replace('\n', ' ')

                # 2a. Cache check
                cached_ans = redis_client.get(f"stealth:ans:{q_hash}")
                if cached_ans:
                    try:
                        parsed_ans = json.loads(cached_ans)
                        results_output[i] = parsed_ans
                        completed += 1
                        logger.info(f"  [CACHE] #{i+1}")
                        yield f"data: {json.dumps({'type': 'progress', 'ready': completed, 'total': total})}\n\n"
                        continue
                    except:
                        pass

                lock_key = f"stealth:pending:{q_hash}"
                lock_acquired = redis_client.set(lock_key, "1", nx=True, ex=300)

                if lock_acquired:
                    task = celery_client.send_task('stealth_tasks.process_single_question', args=[i, q_text, base_prompt])
                    redis_client.set(lock_key, task.id, xx=True, ex=300)
                    owned_tasks[i] = {"task_id": task.id, "q_text": q_text, "hash": q_hash}
                    logger.info(f"  [SEND]  #{i+1}")
                else:
                    subscribed_tasks[i] = {"q_text": q_text, "hash": q_hash}
                    logger.info(f"  [WAIT]  #{i+1} | {q_short}... (MD5:{q_hash[:8]}, other solving)")

            all_pending = len(owned_tasks) + len(subscribed_tasks)
            logger.info(f"{'~'*50}")
            logger.info(f"Total: cached={completed} | sent={len(owned_tasks)} | subscribed={len(subscribed_tasks)}")
            logger.info(f"")
            logger.info(f"[Step 3] Polling {all_pending} tasks...")

            pubsub = None
            try:
                # Subscribe to all pending question channels via Pub/Sub
                all_hashes = set()
                for i, t_info in owned_tasks.items():
                    all_hashes.add(t_info['hash'])
                for i, s_info in subscribed_tasks.items():
                    all_hashes.add(s_info['hash'])

                if all_hashes:
                    pubsub = redis_pubsub_pool.pubsub()
                    channels = [f"stealth:solved:{h}" for h in all_hashes]
                    pubsub.subscribe(*channels)

                while owned_tasks or subscribed_tasks:
                    done_owned = []
                    done_subscribed = []

                    # Wait for Pub/Sub notification (up to 0.5s)
                    if pubsub:
                        pubsub.get_message(timeout=0.5)

                    # 3a. Check OWNED Celery tasks
                    for i, t_info in owned_tasks.items():
                        res = celery_client.AsyncResult(t_info["task_id"])
                        if res.ready():
                            try:
                                answer_json = res.get()
                                if answer_json:
                                    results_output[i] = answer_json
                                    if "Error" not in str(answer_json.get("options", [])):
                                        redis_client.set(f"stealth:ans:{t_info['hash']}", json.dumps(answer_json), ex=1800)
                                        redis_client.delete(f"stealth:pending:{t_info['hash']}")
                                        # Notify subscribers via Pub/Sub
                                        redis_pubsub_pool.publish(f"stealth:solved:{t_info['hash']}", "1")
                                        correct = answer_json.get('correct_index', '?')
                                        logger.info(f"  [OK] #{i+1} SOLVED | correct_index={correct} (cached+published)")
                                    else:
                                        redis_client.delete(f"stealth:pending:{t_info['hash']}")
                                        redis_pubsub_pool.publish(f"stealth:solved:{t_info['hash']}", "1")
                                        logger.warning(f"  [!] #{i+1} ERROR | {answer_json.get('options', ['?'])[0][:60]}")
                                else:
                                    results_output[i] = {"question": t_info["q_text"], "options": ["Error parsing AI response"], "correct_index": 0}
                                    redis_client.delete(f"stealth:pending:{t_info['hash']}")
                                    redis_pubsub_pool.publish(f"stealth:solved:{t_info['hash']}", "1")
                                    logger.error(f"  [FAIL] #{i+1} EMPTY | Celery returned None")
                            except Exception as exc:
                                results_output[i] = {"question": t_info["q_text"], "options": [f"Error: {exc}"], "correct_index": 0}
                                redis_client.delete(f"stealth:pending:{t_info['hash']}")
                                redis_pubsub_pool.publish(f"stealth:solved:{t_info['hash']}", "1")
                                logger.error(f"  [FAIL] #{i+1} EXCEPT | {exc}")

                            done_owned.append(i)
                            completed += 1
                            yield f"data: {json.dumps({'type': 'progress', 'ready': completed, 'total': total})}\n\n"

                    # 3b. Check SUBSCRIBED tasks (woken by Pub/Sub)
                    for i, s_info in subscribed_tasks.items():
                        cached_ans = redis_client.get(f"stealth:ans:{s_info['hash']}")
                        if cached_ans:
                            try:
                                parsed_ans = json.loads(cached_ans)
                                results_output[i] = parsed_ans
                                done_subscribed.append(i)
                                completed += 1
                                logger.info(f"  [CACHE] #{i+1}")
                                yield f"data: {json.dumps({'type': 'progress', 'ready': completed, 'total': total})}\n\n"
                            except:
                                pass
                        else:
                            lock_exists = redis_client.exists(f"stealth:pending:{s_info['hash']}")
                            if not lock_exists:
                                relock = redis_client.set(f"stealth:pending:{s_info['hash']}", "1", nx=True, ex=300)
                                if relock:
                                    task = celery_client.send_task('stealth_tasks.process_single_question', args=[i, s_info["q_text"], base_prompt])
                                    redis_client.set(f"stealth:pending:{s_info['hash']}", task.id, xx=True, ex=300)
                                    owned_tasks[i] = {"task_id": task.id, "q_text": s_info["q_text"], "hash": s_info["hash"]}
                                    done_subscribed.append(i)
                                    # Subscribe to new channel for this task
                                    if pubsub:
                                        pubsub.subscribe(f"stealth:solved:{s_info['hash']}")
                                    logger.warning(f"  [!] #{i+1} TAKEOVER | owner disconnected")

                    for k in done_owned:
                        del owned_tasks[k]
                    for k in done_subscribed:
                        del subscribed_tasks[k]

                # 4. Final results
                final_answers = [ans for ans in results_output if ans is not None]
                total_time = time.time() - t_start
                logger.info(f"")
                logger.info(f"==================================================")
                logger.info(f"[OK] TEST DONE! {len(final_answers)}/{total} answers in {total_time:.1f}s")
                logger.info(f"==================================================")
                yield f"data: {json.dumps({'type': 'result', 'answers': final_answers})}\n\n"

            except GeneratorExit:
                logger.warning(f"[!] Client disconnected! Cancelling {len(owned_tasks)} owned tasks...")
                for i, t_info in owned_tasks.items():
                    try:
                        celery_client.control.revoke(t_info["task_id"], terminate=True)
                        redis_client.delete(f"stealth:pending:{t_info['hash']}")
                    except: pass
                raise
            finally:
                # Cleanup: close pubsub subscription
                if pubsub:
                    try:
                        pubsub.unsubscribe()
                        pubsub.close()
                    except:
                        pass
                # Cleanup: revoke our owned tasks that haven't finished
                for i, t_info in owned_tasks.items():
                    try:
                        res = celery_client.AsyncResult(t_info["task_id"])
                        if not res.ready():
                            celery_client.control.revoke(t_info["task_id"], terminate=True)
                            redis_client.delete(f"stealth:pending:{t_info['hash']}")
                    except: pass

        except GeneratorExit:
            pass
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    resp = Response(stream_with_context(generate()), content_type='text/event-stream')
    resp.headers['Cache-Control'] = 'no-cache'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp

@app.route("/api/tests/export", methods=["GET"])
def export_tests():
    """Export all saved tests as a ZIP archive of .txt files."""
    tests = get_all_tests()
    if not tests:
        return jsonify({"success": False, "error": "No tests found"}), 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for test_id, submitted_at, raw_text in tests:
            # Format: test_001_2026-03-10_12-30-00.txt
            safe_date = submitted_at.replace(':', '-').replace(' ', '_')
            filename = f"test_{test_id:03d}_{safe_date}.txt"
            zf.writestr(filename, raw_text)

    buf.seek(0)
    return send_file(
        buf,
        mimetype='application/zip',
        as_attachment=True,
        download_name='all_tests.zip'
    )

if __name__ == "__main__":
    from waitress import serve
    print(">> Starting Waitress server on http://0.0.0.0:5000", flush=True)
    serve(app, host="0.0.0.0", port=5000, threads=100, channel_timeout=300)
