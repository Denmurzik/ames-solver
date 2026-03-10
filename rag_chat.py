import requests
import json
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Настройки API (Local Proxy)
API_KEY = "sk-Hou2Ci2dLLEZblDfTGM30PatSWUq9WhKD1YrHZnK64Boa"
MODEL = "gemini-3-flash"
BASE_URL = "http://127.0.0.1:8317/v1beta/models"

# Настройки БД
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "book_knowledge"

# Настройки поиска (RAG Pipeline)
VECTOR_SEARCH_TOP_K = 30   # 1 ЭТАП: Достаем 30 кусков из базы быстрым поиском (Bi-Encoder)
RERANK_TOP_K = 10          # 2 ЭТАП: Оставляем 10 самых идеальных кусков умным поиском (Cross-Encoder)

# Формируем полный URL для потоковой генерации контента
url = f"{BASE_URL}/{MODEL}:streamGenerateContent?key={API_KEY}"
headers = {"Content-Type": "application/json"}

# Инициализируем историю сообщений
chat_history = []

def init_chromadb():
    print("Подключение к локальной базе знаний...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        
        # Должна совпадать с моделью из setup_db.py
        model_name = "intfloat/multilingual-e5-small"
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        
        collection = client.get_collection(
            name=COLLECTION_NAME, 
            embedding_function=sentence_transformer_ef
        )
        return collection
    except Exception as e:
        print(f"Ошибка подключения к базе (возможно, она еще не создана): {e}")
        print("Сначала запустите setup_db.py")
        exit(1)

def init_reranker():
    print("Загрузка Cross-Encoder модели для умной сортировки ответов...")
    # Мультиязычный кросс-энкодер, стабильно доступный без авторизации (mMiniLMv2)
    reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    return reranker

def query_gemini(system_prompt, user_query, chat_context):
    full_prompt = f"""{system_prompt}\n\nТекущий вопрос: {user_query}"""
    
    history_with_context = list(chat_context)
    history_with_context.append({"role": "user", "parts": [{"text": full_prompt}]})

    data = {
        "contents": history_with_context
    }

    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        if response.status_code != 200:
            print(f"\n[Ошибка API {response.status_code}]: {response.text}")
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
                            chunk_text = parts[0]["text"]
                            print(chunk_text, end="", flush=True)
                            full_assistant_response += chunk_text
                except json.JSONDecodeError:
                    pass
        print("\n")
        return full_assistant_response
        
    except Exception as e:
        print(f"\n[Критическая ошибка]: {e}\n")
        return None

def query_gemini_silent(system_prompt, user_query, max_retries=5):
    full_prompt = f"{system_prompt}\n\nТекущий вопрос: {user_query}"
    
    data = {"contents": [{"role": "user", "parts": [{"text": full_prompt}]}]}

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, stream=True)
            if response.status_code == 503:
                delay = 2 ** attempt
                print(f"\n[Сервер перегружен, ожидание {delay}с...] Попытка {attempt+1}/{max_retries}")
                time.sleep(delay)
                continue
                
            if response.status_code != 200:
                print(f"\n[Ошибка API {response.status_code}]: {response.text}")
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
                    except json.JSONDecodeError:
                        pass
            return full_assistant_response
            
        except Exception as e:
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                print(f"\n[Ошибка соединения: {e}] Повтор через {delay}с...")
                time.sleep(delay)
                continue
            print(f"\n[Критическая ошибка]: {e}\n")
            return None
    
    return None

def extract_questions_with_gemini(text):
    print("[Умный парсинг] ИИ анализирует структуру теста и выделяет вопросы...")
    system_prompt = '''Ты помощник-машина. Твоя единственная цель — вытащить из предоставленного сырого текста отдельные вопросы (задачи) теста вместе с их вариантами ответов (если они есть).
Верни результат СТРОГО в формате валидного JSON-массива строк. Каждая строка — это полный текст одного вопроса (и всех его вариантов).
Пример: ["Текст первого вопроса и варианты...", "Текст второго вопроса..."]
НИКАКИХ комментариев, пояснений или Markdown разметки. Возвращай только JSON, начинающийся с [ и заканчивающийся на ].'''
    
    response = query_gemini_silent(system_prompt, text)
    if not response:
        return []
    
    try:
        cleaned = response.strip()
        if cleaned.startswith("```json"): cleaned = cleaned[7:]
        if cleaned.startswith("```"): cleaned = cleaned[3:]
        if cleaned.endswith("```"): cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        questions = json.loads(cleaned)
        if isinstance(questions, list):
            return questions
    except Exception as e:
        print(f"\n[Ошибка ИИ-парсера]: Не удалось распознать формат массива. Ответ ИИ:\n{response[:200]}...\n")
        
    return []

def process_single_question(i, question_text, collection, reranker, base_prompt):
    try:
        formatted_query = f"query: {question_text}"
        results = collection.query(query_texts=[formatted_query], n_results=VECTOR_SEARCH_TOP_K)
        
        retrieved_chunks = results['documents'][0] if results['documents'] else []
        retrieved_metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        context_text = ""
        if retrieved_chunks:
            clean_chunks = [chunk.replace("passage: ", "", 1) if chunk.startswith("passage: ") else chunk for chunk in retrieved_chunks]
            pairs = [[question_text, chunk] for chunk in clean_chunks]
            scores = reranker.predict(pairs)
            scored_results = list(zip(clean_chunks, retrieved_metadatas, scores.tolist() if hasattr(scores, 'tolist') else scores))
            scored_results.sort(key=lambda x: x[2], reverse=True)
            best_results = scored_results[:RERANK_TOP_K]
            
            context_parts = []
            for idx, (text, meta, score) in enumerate(best_results):
                page_num = meta.get("page", "?")
                context_parts.append(f"--- ФРАГМЕНТ {idx+1} [Страница: {page_num}] ---\n{text}")
            context_text = "\n\n".join(context_parts)
            
        system_prompt = f"{base_prompt}\n\nИНФОРМАЦИЯ ИЗ КНИГИ:\n{context_text}"
        
        assistant_response = query_gemini_silent(system_prompt, question_text)
        
        if assistant_response:
            return i, assistant_response.strip()
        else:
            return i, f"Ошибка при обработке вопроса {i+1}"
    except Exception as e:
        return i, f"Критическая ошибка при обработке вопроса {i+1}: {e}"

def process_test_file(collection, reranker, test_filepath="test.txt", max_workers=3):
    if not os.path.exists(test_filepath):
        print(f"[Ошибка] Файл {test_filepath} не найден. Создайте его и добавьте туда текст теста.")
        return

    print(f"\n[Запуск режима тестирования. Чтение файла {test_filepath}...]")
    with open(test_filepath, "r", encoding="utf-8") as f:
        content = f.read()

    questions = extract_questions_with_gemini(content)

    if not questions:
        print("[Ошибка] Не удалось распознать вопросы. Убедитесь, что текст содержит вопросы.")
        return

    print(f"ИИ успешно нашел {len(questions)} вопросов. Начинаю ПАРАЛЛЕЛЬНЫЙ поиск и генерацию...")
    
    base_prompt = "Ты — ИИ-ассистент, реши тест."
    if os.path.exists("prompt.txt"):
        try:
            with open("prompt.txt", "r", encoding="utf-8") as f:
                p_content = f.read().strip()
                if p_content: base_prompt = p_content
        except: pass
    
    # Подготавливаем массив для сохранения оригинального порядка ответов
    results_output = [""] * len(questions)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Запускаем все вопросы в параллель
        future_to_index = {
            executor.submit(process_single_question, i, q_text, collection, reranker, base_prompt): i 
            for i, q_text in enumerate(questions)
        }
        
        completed_count = 0
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                _, answer = future.result()
                results_output[i] = answer
                completed_count += 1
                print(f"Готово вопросов: {completed_count}/{len(questions)}...", end="\r")
            except Exception as exc:
                results_output[i] = f"Ошибка в потоке: {exc}"
                completed_count += 1
             
    print(f"\n[Завершено] Все потоки отработали. Сохраняю в test_results.txt...               ")
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(results_output))
        
    print("Готово! Результаты сохранены в файл test_results.txt\n")

def main():
    print("\n=== RAG Чат v2.0 запущен (Antigravity Proxy API) ===")
    collection = init_chromadb()
    reranker = init_reranker()
    
    print(f"База знаний успешно загружена. Количество записей: {collection.count()}")
    print("Включен режим Продвинутого Поиска (Bi-Encoder + Cross-Encoder Reranking)")
    print("Введите /test для автоматического решения всех вопросов из test.txt")
    print("Введите 'выход', 'exit' или 'quit' для завершения.\n")

    while True:
        try:
            user_input = input("Вы: ")
            
            if user_input.strip().lower() in ['выход', 'exit', 'quit']:
                print("Завершение работы.")
                break
                
            if user_input.strip() == "/test":
                process_test_file(collection, reranker)
                continue
                
            if not user_input.strip():
                continue

            print("\n[ЭТАП 1: Быстрый поиск топ-15 результатов в VDB...]", end="\r")
            
            # В мультиязычной E5 модели запросы должны начинаться с "query: " 
            formatted_query = f"query: {user_input}"
            
            results = collection.query(
                query_texts=[formatted_query],
                n_results=VECTOR_SEARCH_TOP_K
            )
            
            retrieved_chunks = results['documents'][0] if results['documents'] else []
            retrieved_metadatas = results['metadatas'][0] if results['metadatas'] else []
            
            if not retrieved_chunks:
                context_text = ""
                print("[В базе знаний ничего не найдено, ответ из общих знаний]")
            else:
                print(f"[ЭТАП 2: Умное переранжирование (Cross-Encoder) {len(retrieved_chunks)} кусков...]", end="\r")
                
                # Подготавливаем пары "Запрос - Кусок текста" для реранкера
                clean_chunks = [chunk.replace("passage: ", "", 1) if chunk.startswith("passage: ") else chunk for chunk in retrieved_chunks]
                pairs = [[user_input, chunk] for chunk in clean_chunks]
                
                # Вычисляем истинную оценку релевантности для каждой пары через CrossEncoder
                scores = reranker.predict(pairs)
                
                # Сортируем куски по убыванию оценки
                # scores - numpy array, нужно конвертировать для работы с zip
                scored_results = list(zip(clean_chunks, retrieved_metadatas, scores.tolist() if hasattr(scores, 'tolist') else scores))
                scored_results.sort(key=lambda x: x[2], reverse=True)
                
                # Берем только топ-N самых лучших
                best_results = scored_results[:RERANK_TOP_K]
                
                # Формируем контекст с метаданными (Страница)
                context_parts = []
                for idx, (text, meta, score) in enumerate(best_results):
                    page_num = meta.get("page", "?")
                    # Добавляем номер фрагмента и страницу для Gemini
                    context_parts.append(f"--- ФРАГМЕНТ {idx+1} [Страница: {page_num}] ---\n{text}")
                    
                context_text = "\n\n".join(context_parts)
                print(f"[Контекст сформирован из {len(best_results)} лучших страниц. Генерация ответа...]     ")
            
            # Читаем кастомный промпт
            base_prompt = "Ты — интеллектуальный ИИ-ассистент, эксперт по анализу данных. Ниже приведена информация (фрагменты из книги) с указанием страниц. Пожалуйста, ответь на вопрос пользователя, опираясь ПРЕИМУЩЕСТВЕННО на эти фрагменты. Если уместно, укажи страницу, откуда взята информация. Если в фрагментах нет прямого ответа, попытайся ответить, используя свои общие знания, но обязательно предупреди, что в текущем контексте книги этого нет."
            
            if os.path.exists("prompt.txt"):
                try:
                    with open("prompt.txt", "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            base_prompt = content
                except: pass

            if context_text:
                system_prompt = f"{base_prompt}\n\nИНФОРМАЦИЯ ИЗ КНИГИ:\n{context_text}"
            else:
                system_prompt = base_prompt

            print("Gemini: ", end="", flush=True)

            assistant_response = query_gemini(system_prompt, user_input, chat_history)

            if assistant_response:
                chat_history.append({"role": "user", "parts": [{"text": user_input}]})
                chat_history.append({"role": "model", "parts": [{"text": assistant_response}]})

        except KeyboardInterrupt:
            print("\nЗавершение работы.")
            break
        except Exception as e:
            print(f"\n[Ошибка]: {e}\n")

if __name__ == "__main__":
    main()
