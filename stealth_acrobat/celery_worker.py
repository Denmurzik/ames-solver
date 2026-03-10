import os
import tempfile
import json
import requests
import time
import hashlib
import pickle
import redis as redis_lib
import re
import itertools
import threading
from celery import Celery
import chromadb
from sentence_transformers import SentenceTransformer
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from celery.signals import worker_process_init
import logging

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

celery_app = Celery('stealth_tasks', broker='redis://127.0.0.1:6379/0', backend='redis://127.0.0.1:6379/1')

API_KEY = "sk-Hou2Ci2dLLEZblDfTGM30PatSWUq9WhKD1YrHZnK64Boa"
MODEL = "gemini-3-flash"
GEMINI_ENDPOINTS = [
    f"http://127.0.0.1:8317/v1beta/models/{MODEL}:streamGenerateContent?key={API_KEY}",
    f"http://127.0.0.1:8318/v1beta/models/{MODEL}:streamGenerateContent?key={API_KEY}",
    f"http://127.0.0.1:8319/v1beta/models/{MODEL}:streamGenerateContent?key={API_KEY}",
]
# Circuit Breaker для Gemini прокси
class GeminiCircuitBreaker:
    def __init__(self, endpoints, fail_threshold=3, recovery_timeout=60):
        self._endpoints = list(endpoints)
        self._fail_threshold = fail_threshold
        self._recovery_timeout = recovery_timeout
        self._failures = {ep: 0 for ep in endpoints}
        self._open_until = {ep: 0.0 for ep in endpoints}
        self._idx = 0
        self._lock = threading.Lock()

    def get_endpoint(self):
        with self._lock:
            now = time.time()
            tried = 0
            while tried < len(self._endpoints):
                ep = self._endpoints[self._idx % len(self._endpoints)]
                self._idx += 1
                tried += 1
                if self._open_until[ep] <= now:
                    return ep
            # All open -> fallback to earliest recovery
            earliest = min(self._endpoints, key=lambda e: self._open_until[e])
            logger.warning(f"[CB] All endpoints open! Fallback to {earliest.split(':')[2].split('/')[0]}")
            return earliest

    def report_success(self, endpoint):
        with self._lock:
            if self._failures[endpoint] > 0:
                port = endpoint.split(':')[2].split('/')[0]
                logger.info(f"[CB] :{port} recovered — back in rotation")
            self._failures[endpoint] = 0
            self._open_until[endpoint] = 0.0

    def report_failure(self, endpoint):
        with self._lock:
            self._failures[endpoint] += 1
            if self._failures[endpoint] >= self._fail_threshold:
                self._open_until[endpoint] = time.time() + self._recovery_timeout
                port = endpoint.split(':')[2].split('/')[0]
                logger.warning(f"[CB] :{port} OPEN — excluded for {self._recovery_timeout}s ({self._failures[endpoint]} failures)")

_gemini_cb = GeminiCircuitBreaker(GEMINI_ENDPOINTS)

headers = {"Content-Type": "application/json"}

# HTTP connection pool (keep-alive per Gemini endpoint)
_http_session = requests.Session()
_http_session.headers.update(headers)

# Redis client for embedding cache (db=2, same as app.py)
_embed_redis = redis_lib.Redis(host='127.0.0.1', port=6379, db=2, decode_responses=False)

# RAG Configuration
CHROMA_DB_DIR = "/home/denis/ames-suck-my-ass/chroma_db"
COLLECTION_NAME = "book_knowledge"
EMBED_MODEL_NAME = "perplexity-ai/pplx-embed-context-v1-0.6B"
ONNX_RERANKER_DIR = "/home/denis/ames-suck-my-ass/onnx_reranker"
BM25_INDEX_PATH = "/home/denis/ames-suck-my-ass/bm25_index.pkl"
VECTOR_SEARCH_TOP_K = 20
HYBRID_TOP_K = 20
RRF_K = 60  # Reciprocal Rank Fusion constant

collection = None
reranker = None
embed_model = None
_bm25_data = None  # BM25 index data

_ml_lock = threading.Lock()
_ml_semaphore = threading.Semaphore(3)  # Max 3 concurrent ML inferences on 4 CPU cores

# Embedding Batching Configuration
_batch_queue = []
_batch_lock = threading.Lock()
_batch_event = threading.Event()

def _process_batches_loop():
    global embed_model, collection
    while True:
        _batch_event.wait(timeout=0.1) # Wait up to 100ms or until signaled
        _batch_event.clear()
        
        with _batch_lock:
            if not _batch_queue:
                continue
            batch = _batch_queue.copy()
            _batch_queue.clear()
            
        if not batch: continue
        
        try:
            if embed_model is None or collection is None:
                logger.warning("[BATCH] Models not ready yet, signaling error")
                for _, event, result_dict in batch:
                    result_dict['error'] = True
                    event.set()
                continue
            texts = [item[0] for item in batch]
            logger.info(f"   [BATCH] Running encode for {len(texts)} questions")
            t_start = time.time()
            embeddings = embed_model.encode(texts, normalize_embeddings=True, batch_size=len(texts))
            # Run ChromaDB search in batch too for even better performance
            results = collection.query(query_embeddings=embeddings.tolist(), n_results=VECTOR_SEARCH_TOP_K)
            bot_t = time.time() - t_start
            logger.info(f"   [BATCH] Finished {len(texts)} in {bot_t:.2f}s")
            
            for i, (_, event, result_dict) in enumerate(batch):
                result_dict['embedding'] = embeddings[i].tolist()
                result_dict['chunks'] = results['documents'][i] if results['documents'] else []
                result_dict['metadatas'] = results['metadatas'][i] if results['metadatas'] else []
                event.set()
        except Exception as e:
            logger.error(f"[BATCH] Error processing batch: {e}")
            for _, event, result_dict in batch:
                result_dict['error'] = True
                event.set()

_batch_thread = threading.Thread(target=_process_batches_loop, daemon=True)
_batch_thread_started = False

def get_rag_models():
    global collection, reranker, embed_model
    if collection is None or reranker is None or embed_model is None:
        with _ml_lock:
            # Double-checked locking
            if collection is None or reranker is None or embed_model is None:
                try:
                    t0 = time.time()
                    logger.info(f"\n{'='*60}")
                    logger.info(f">> INIT RAG PIPELINE")
                    logger.info(f"{'='*60}")
                    
                    logger.info(f"[1/3] Loading Embedding: {EMBED_MODEL_NAME}")
                    embed_model = SentenceTransformer(EMBED_MODEL_NAME, trust_remote_code=True)
                    logger.info(f"   [OK] Embedding loaded ({time.time()-t0:.1f}s)")

                    t1 = time.time()
                    logger.info(f"[2/3] Connecting ChromaDB: {CHROMA_DB_DIR}")
                    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
                    collection = client.get_collection(name=COLLECTION_NAME)
                    logger.info(f"   [OK] DB: {collection.count()} records ({time.time()-t1:.1f}s)")
                    
                    reranker = "DISABLED"
                    
                    # Load BM25 index
                    global _bm25_data
                    if os.path.exists(BM25_INDEX_PATH):
                        t3 = time.time()
                        logger.info(f"[3/3] Loading BM25 index: {BM25_INDEX_PATH}")
                        with open(BM25_INDEX_PATH, 'rb') as bf:
                            _bm25_data = pickle.load(bf)
                        logger.info(f"   [OK] BM25 loaded: {len(_bm25_data['corpus_texts'])} docs ({time.time()-t3:.1f}s)")
                    else:
                        logger.warning(f"[!] BM25 index not found at {BM25_INDEX_PATH}")
                    
                    logger.info(f"{'='*60}")
                    
                    global _batch_thread_started
                    if not _batch_thread_started:
                        _batch_thread.start()
                        _batch_thread_started = True
                    logger.info(f"[OK] RAG PIPELINE READY! (total: {time.time()-t0:.1f}s)")
                    logger.info(f"   Embedding: {EMBED_MODEL_NAME}")
                    logger.info(f"   ChromaDB: {collection.count()} records")
                    logger.info(f"   TOP_K: search={VECTOR_SEARCH_TOP_K}, rerank={RERANK_TOP_K}")
                    logger.info(f"{'='*60}\n")
                except Exception as e:
                    logger.error(f"[FAIL] RAG init error: {e}")
    return collection, reranker, embed_model

def query_gemini_silent(system_prompt, user_query, max_retries=5):
    full_prompt = f"{system_prompt}\n\nТекущий вопрос: {user_query}"
    data = {"contents": [{"role": "user", "parts": [{"text": full_prompt}]}]}

    for attempt in range(max_retries):
        selected_url = _gemini_cb.get_endpoint()
        port = selected_url.split(':')[2].split('/')[0]
        try:
            t_api = time.time()
            response = _http_session.post(selected_url, json=data, stream=True, timeout=120)
            if response.status_code in [429, 503, 502, 504]:
                _gemini_cb.report_failure(selected_url)
                delay = (2 ** attempt) + 1
                logger.warning(f"   [!] Gemini (:{port}) HTTP {response.status_code} -- retry {attempt+1}/{max_retries} in {delay}s")
                time.sleep(delay)
                continue
                
            if response.status_code != 200:
                _gemini_cb.report_failure(selected_url)
                logger.error(f"   [FAIL] Gemini (:{port}) HTTP {response.status_code}: {response.text[:200]}")
                return None

            full_assistant_response = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        decoded_line = decoded_line[6:]
                    if decoded_line.strip() in ["[", "]", ","]: 
                        continue
                    try:
                        if decoded_line.endswith(","):
                            decoded_line = decoded_line[:-1]
                        json_data = json.loads(decoded_line)
                        if "candidates" in json_data and len(json_data["candidates"]) > 0:
                            content = json_data["candidates"][0].get("content", {})
                            parts = content.get("parts", [])
                            if parts and "text" in parts[0]:
                                full_assistant_response += parts[0]["text"]
                    except json.JSONDecodeError: pass
            api_time = time.time() - t_api
            _gemini_cb.report_success(selected_url)
            logger.info(f"   [OK] Gemini (:{port}) responded in {api_time:.1f}s ({len(full_assistant_response)} chars)")
            return full_assistant_response
        except requests.exceptions.Timeout:
            _gemini_cb.report_failure(selected_url)
            logger.warning(f"   [!] Gemini (:{port}) TIMEOUT -- retry {attempt+1}/{max_retries}")
            continue
        except Exception as e:
            _gemini_cb.report_failure(selected_url)
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                logger.warning(f"   [!] Gemini (:{port}) error: {e} -- retry {attempt+1}/{max_retries} in {delay}s")
                time.sleep(delay)
                continue
            logger.error(f"   [FAIL] Gemini (:{port}) all retries exhausted: {e}")
            return None
    return None

def extract_json_object(raw_text):
    if not raw_text:
        return None
    ans = raw_text.strip()
    if ans.startswith("```json"): ans = ans[7:]
    if ans.startswith("```"): ans = ans[3:]
    if ans.endswith("```"): ans = ans[:-3]
    ans = ans.strip()
    try:
        return json.loads(ans)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', ans, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None

@celery_app.task(name='stealth_tasks.process_single_question', bind=True)
def process_single_question_task(self, i, question_text, base_prompt):
    q_preview = question_text[:80].replace('\n', ' ')
    logger.info(f"")
    logger.info(f"--------------------------------------------------")
    logger.info(f"[Q] QUESTION #{i+1}: {q_preview}...")
    logger.info(f"--------------------------------------------------")
    
    t_total = time.time()
    col, rerank, emb_model = get_rag_models()
    if not col or not emb_model:
        logger.error(f"   [FAIL] RAG models not loaded!")
        return {"question": question_text, "options": ["Error: RAG models missing"], "correct_index": 0}
        
    try:
        # --- ML INFERENCE (CPU-bound, throttled by semaphore) ---
        with _ml_semaphore:
            # Шаг 1+2: Embedding + ChromaDB (с кэшированием)
            t_embed = time.time()
            q_hash = _order_independent_hash(question_text)
            embed_cache_key = f"stealth:emb:{q_hash}".encode()

            cached = _embed_redis.get(embed_cache_key)
            cache_hit = False
            if cached:
                try:
                    cache_data = pickle.loads(cached)
                    query_embedding = cache_data['embedding']
                    retrieved_chunks = cache_data['chunks']
                    retrieved_metadatas = cache_data['metadatas']
                    embed_time = time.time() - t_embed
                    logger.info(f"   [EMBED+SEARCH] CACHE HIT {embed_time:.3f}s ({len(retrieved_chunks)} chunks)")
                    cache_hit = True
                except Exception as e:
                    logger.warning(f"   [CACHE] Invalid data, ignoring: {e}")

            if not cache_hit:
                # Add to batch queue and wait
                evt = threading.Event()
                res_dict = {}
                with _batch_lock:
                    _batch_queue.append((question_text, evt, res_dict))
                    if len(_batch_queue) >= 3: # If we have 3, trigger immediately
                        _batch_event.set()
                
                # Wait for background thread to process the batch
                evt.wait(timeout=30.0)
                
                if res_dict.get('error') or 'embedding' not in res_dict:
                    # Fallback to synchronous if batching failed
                    logger.warning("   [BATCH] Fallback to sync encode")
                    query_embedding = emb_model.encode(question_text, normalize_embeddings=True).tolist()
                    results = collection.query(query_embeddings=[query_embedding], n_results=VECTOR_SEARCH_TOP_K)
                    retrieved_chunks = results['documents'][0] if results['documents'] else []
                    retrieved_metadatas = results['metadatas'][0] if results['metadatas'] else []
                else:
                    query_embedding = res_dict['embedding']
                    retrieved_chunks = res_dict['chunks']
                    retrieved_metadatas = res_dict['metadatas']

                # Cache embedding + search results (TTL 10 min)
                try:
                    _embed_redis.set(embed_cache_key, pickle.dumps({
                        'embedding': query_embedding,
                        'chunks': retrieved_chunks,
                        'metadatas': retrieved_metadatas,
                    }), ex=600)
                except Exception:
                    pass  # Non-critical: cache write failure
            
            # Шаг 3: Hybrid Search (Vector + BM25 via RRF)
            context_text = ""
            if retrieved_chunks:
                # Build vector ranking
                vector_ranking = {}  # text -> rank
                vector_meta = {}    # text -> metadata
                for rank, (chunk, meta) in enumerate(zip(retrieved_chunks, retrieved_metadatas)):
                    vector_ranking[chunk] = rank
                    vector_meta[chunk] = meta
                
                # BM25 search
                bm25_ranking = {}
                if _bm25_data:
                    query_tokens = re.findall(r'[а-яёa-z0-9]+', question_text.lower())
                    scores = _bm25_data['bm25'].get_scores(query_tokens)
                    top_indices = scores.argsort()[-VECTOR_SEARCH_TOP_K:][::-1]
                    for rank, idx in enumerate(top_indices):
                        if scores[idx] > 0:
                            txt = _bm25_data['corpus_texts'][idx]
                            bm25_ranking[txt] = rank
                            if txt not in vector_meta:
                                vector_meta[txt] = _bm25_data['corpus_metadatas'][idx]
                
                # RRF merge with source priority boosting
                SOURCE_BOOST = {
                    "pdf24_merged.pdf": 1.2,
                    "Irtegov-OS-Unix-System-Calls.pdf": 1.1,
                    "book.pdf": 1.0,
                }
                all_chunks = set(vector_ranking.keys()) | set(bm25_ranking.keys())
                rrf_scores = []
                for chunk in all_chunks:
                    v_rank = vector_ranking.get(chunk, 1000)
                    b_rank = bm25_ranking.get(chunk, 1000)
                    meta = vector_meta.get(chunk, {})
                    boost = SOURCE_BOOST.get(meta.get('source', ''), 1.0)
                    score = (1.0 / (RRF_K + v_rank) + 1.0 / (RRF_K + b_rank)) * boost
                    rrf_scores.append((chunk, meta, score))
                
                rrf_scores.sort(key=lambda x: x[2], reverse=True)
                best = rrf_scores[:HYBRID_TOP_K]
                
                bm25_hits = len(bm25_ranking)
                top_source = best[0][1].get('source', '?') if best else '?'
                top_page = best[0][1].get('page', '?') if best else '?'
                logger.info(f"   [HYBRID] vec={len(retrieved_chunks)} + bm25={bm25_hits} -> top-{len(best)} (src={top_source} p.{top_page})")
                
                context_parts = []
                for idx, (chunk_text, meta, score) in enumerate(best):
                    page_num = meta.get('page', '?')
                    context_parts.append(f"--- FRAGMENT {idx+1} [Page: {page_num}] ---\n{chunk_text}")
                context_text = "\n\n".join(context_parts)
        # Шаг 4: Gemini API
        system_prompt = f"{base_prompt}\n\nИНФОРМАЦИЯ ИЗ ИСТОЧНИКОВ:\n{context_text}"
        json_prompt = f"""{system_prompt}

ЗАДАЧА: Ответь на тест-вопрос ниже.

ШАГ 1 — ОПРЕДЕЛИ ТИП ВОПРОСА:
• Множественный выбор (чекбоксы, «выберите все верные», «отметьте все», «какие из»): correct_index = [массив].
• Единственный выбор (радиокнопка, «выберите один», «какой», «что является») ИЛИ тип неясен: correct_index = число.
• Свободный ввод (нет вариантов): запиши ответ в options[0], correct_index = 0.

ШАГ 2 — ДЛЯ МНОЖЕСТВЕННОГО ВЫБОРА — проанализируй КАЖДЫЙ вариант:
[АНАЛИЗ]
#0 «текст» — ✓/✗ (кратко почему)
#1 «текст» — ✓/✗ (кратко почему)
...
[/АНАЛИЗ]

ШАГ 3 — ФИНАЛЬНАЯ ПРОВЕРКА (только для множественного выбора):
Пересмотри каждый вариант, помеченный ✓. Спроси себя: «Есть ли ПРЯМОЕ подтверждение в источниках?»
Если подтверждения нет или оно косвенное — убери ✓. Лучше не отметить один правильный, чем отметить лишний неправильный.

ФОРМАТ ОТВЕТА — строго JSON:
{{
  "question": "Текст вопроса",
  "options": ["Вариант 1", "Вариант 2", "Вариант 3"],
  "correct_index": 0
}}

ПРАВИЛА:
- options ОБЯЗАНО содержать ВСЕ варианты из вопроса (и правильные, и неправильные).
- Для единственного выбора: correct_index = число.
- Для множественного выбора: correct_index = [массив чисел].

ПРИМЕРЫ:
Один правильный: {{"question": "Что означает слово “архитектура” в словосочетании “архитектура x86”?", "options": ["Технологический процесс изготовления микросхемы, например, фотолитография 15нм", "Документ, описывающий допустимые команды процессора, их формат
и семантику"], "correct_index": 1}}
Несколько правильных: {{"question": "Для чего обычно используется основная (оперативная) память в типичном современном
компьютере? Выберите все верные ответы.", "options": ["Для хранения машинного кода", "Для хранения переменных языка C, в том числе структур и массивов", "Для передачи параметров в функции", "Для хранения часто используемых и/или промежуточных значений", "Для хранения стека вызовов", "Для управления работой процессора, например для установки системного или
пользовательского режима", "Для хранения файлов", "Для хранения адреса текущей команды"], "correct_index": [0, 1, 2, 4]}}
Свободный ввод: {{"question": "Перечислите названия сегментов, создаваемых по умолчанию в образе процесса. Напишите названия большими латинскими буквами через запятую, никаких других слов набирать не следует.", "options": ["TEXT, DATA, BSS, STACK"], "correct_index": 0}}"""
        for attempt in range(2):
            logger.info(f"   [GEMINI] attempt {attempt+1}/2...")
            assistant_response = query_gemini_silent(json_prompt, question_text)
            if assistant_response:
                parsed = extract_json_object(assistant_response)
                if parsed and isinstance(parsed, dict) and 'correct_index' in parsed:
                    if 'question' not in parsed:
                        parsed['question'] = question_text
                    if 'options' not in parsed:
                        parsed['options'] = []
                    
                    total_time = time.time() - t_total
                    correct = parsed['correct_index']
                    logger.info(f"   [OK] ANSWER: correct_index={correct} (total: {total_time:.1f}s)")
                    return parsed
                else:
                    logger.warning(f"   [!] Gemini replied but JSON parse failed (attempt {attempt+1})")
            else:
                logger.warning(f"   [!] Gemini returned empty (attempt {attempt+1})")
        
        total_time = time.time() - t_total
        logger.error(f"   [FAIL] No answer for Q#{i+1} ({total_time:.1f}s)")
        return {"question": question_text, "options": ["Error parsing AI response (no valid JSON)"], "correct_index": 0}
    except Exception as e:
        total_time = time.time() - t_total
        logger.error(f"   [FAIL] Exception Q#{i+1}: {e} ({total_time:.1f}s)")
        return {"question": question_text, "options": [f"Error: {str(e)}"], "correct_index": 0}

@celery_app.task(name='stealth_tasks.extract_questions', bind=True)
def extract_questions_task(self, text):
    logger.info(f"")
    logger.info(f"==================================================")
    logger.info(f">> NEW REQUEST: extracting questions ({len(text)} chars)")
    logger.info(f"==================================================")
    t0 = time.time()
    system_prompt = '''Ты помощник-машина. Твоя единственная цель — вытащить из предоставленного сырого текста отдельные вопросы (задачи) теста вместе с их вариантами ответов (если они есть).
ВАЖНО: Убери ВСЕ маркеры вариантов ответа (❑, ☐, ☑, a), b), 1), 2), -, *, •, и любые другие). Каждый вариант должен быть на отдельной строке без маркера, только чистый текст.
Верни результат СТРОГО в формате валидного JSON-массива строк. Каждая строка — это полный текст одного вопроса (заголовок + варианты через \\n).
Пример: ["Какой протокол?\\nHTTP\\nSMTP\\nFTP", "Что делает fork?\\nСоздаёт поток\\nСоздаёт процесс"]
НИКАКИХ комментариев, пояснений или Markdown разметки. Возвращай только JSON, начинающийся с [ и заканчивающийся на ].'''
    logger.info(f"   [GEMINI] Sending text for question extraction...")
    response = query_gemini_silent(system_prompt, text)
    if not response:
        logger.error(f"   [FAIL] Gemini returned no response for extraction")
        return []
    try:
        cleaned = response.strip()
        if cleaned.startswith("```json"): cleaned = cleaned[7:]
        if cleaned.startswith("```"): cleaned = cleaned[3:]
        if cleaned.endswith("```"): cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        try:
            questions = json.loads(cleaned)
            if isinstance(questions, list):
                logger.info(f"   [OK] Extracted {len(questions)} questions in {time.time()-t0:.1f}s")
                for qi, q in enumerate(questions):
                    q_short = str(q)[:60].replace('\n', ' ')
                    logger.info(f"      #{qi+1}: {q_short}...")
                return questions
        except json.JSONDecodeError:
            pass
            
        # Fallback regex for array
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if match:
            questions = json.loads(match.group())
            if isinstance(questions, list):
                logger.info(f"   [OK] Extracted {len(questions)} questions (fallback regex) in {time.time()-t0:.1f}s")
                return questions
            
    except Exception as e:
        logger.error(f"   [FAIL] Parser error: {e}, raw: {response[:200]}")
    return []
