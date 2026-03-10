import os
import chromadb
import fitz  # PyMuPDF
import hashlib
import pickle
import re
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ============================================================
# Настройки
# ============================================================
PDF_PATHS = [
    "/home/denis/ames-suck-my-ass/book.pdf",
    "/home/denis/ames-suck-my-ass/Irtegov-OS-Unix-System-Calls.pdf",
    "/home/denis/ames-suck-my-ass/pdf24_merged.pdf",
]
CHROMA_DB_DIR = "/home/denis/ames-suck-my-ass/chroma_db"
COLLECTION_NAME = "book_knowledge"
BM25_INDEX_PATH = "/home/denis/ames-suck-my-ass/bm25_index.pkl"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400

# Контекстная модель Perplexity
EMBED_MODEL_NAME = "perplexity-ai/pplx-embed-context-v1-0.6B"

# Сколько соседних страниц брать для формирования контекста
CONTEXT_WINDOW_PAGES = 1

# Максимальная длина контекста в символах
MAX_CONTEXT_CHARS = 6000

# Размер батча для encode() и сохранения
BATCH_SIZE = 16


def extract_text_from_pdf(pdf_path):
    print(f"📄 Извлечение текста из {os.path.basename(pdf_path)}...")
    doc = fitz.open(pdf_path)
    pages_data = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text").strip()
        if text:
            pages_data.append({
                "text": text,
                "page": page_num + 1,
                "source": os.path.basename(pdf_path)
            })
        if page_num % 100 == 0 and page_num > 0:
            print(f"   Обработано {page_num + 1} страниц из {len(doc)}...")

    print(f"   ✅ [{os.path.basename(pdf_path)}] Извлечено страниц с текстом: {len(pages_data)}.")
    return pages_data


def build_page_context(pages_data, current_idx, window=CONTEXT_WINDOW_PAGES, max_chars=MAX_CONTEXT_CHARS):
    start = max(0, current_idx - window)
    end = min(len(pages_data), current_idx + window + 1)

    context_parts = []
    for i in range(start, end):
        context_parts.append(pages_data[i]["text"])

    context = "\n\n".join(context_parts)

    if len(context) > max_chars:
        current_text = pages_data[current_idx]["text"]
        pos = context.find(current_text)
        if pos == -1:
            context = context[:max_chars]
        else:
            center = pos + len(current_text) // 2
            half_window = max_chars // 2
            ctx_start = max(0, center - half_window)
            ctx_end = min(len(context), center + half_window)
            context = context[ctx_start:ctx_end]

    return context


def chunk_text_with_context(pages_data, chunk_size, chunk_overlap):
    print(f"🔪 Разбиение текста на чанки с контекстом из {pages_data[0]['source']}...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    chunks = []

    for page_idx, page_item in enumerate(pages_data):
        page_text = page_item["text"]
        page_num = page_item["page"]
        source_name = page_item["source"]

        context = build_page_context(pages_data, page_idx)
        page_chunks = text_splitter.split_text(page_text)

        for chunk in page_chunks:
            # Сразу генерируем ID, чтобы потом проверять в БД
            hash_str = f"{source_name}_{page_num}_{hashlib.md5(chunk.encode()).hexdigest()}"
            chunks.append({
                "id": hash_str,
                "text": chunk,
                "context": context,
                "metadata": {"source": source_name, "page": page_num}
            })

    print(f"   ✅ Создано {len(chunks)} фрагментов с контекстом.")
    return chunks


def process_and_save_embeddings(model, chunks, batch_size=BATCH_SIZE):
    print("\n💾 Подключение к ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Коллекция без embedding_function — эмбеддинги вычисляем вручную
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"   В базе сейчас {collection.count()} записей.")
    print("🧠 Начало потоковой векторизации и сохранения...")

    total_encoded = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_ids = [item["id"] for item in batch]
        
        # Проверяем, какие из этих чанков уже есть в базе
        existing_data = collection.get(ids=batch_ids)
        existing_ids = set(existing_data["ids"]) if existing_data and "ids" in existing_data else set()
        
        # Фильтруем те, которых еще нет в БД 
        missing_batch = [item for item in batch if item["id"] not in existing_ids]
        
        if missing_batch:
            # Вычисляем эмбеддинги ТОЛЬКО для недостающих
            pairs = [[item["context"], item["text"]] for item in missing_batch]
            embeddings = model.encode(pairs, normalize_embeddings=True)
            
            missing_ids = [item["id"] for item in missing_batch]
            missing_texts = [item["text"] for item in missing_batch]
            missing_metadatas = [item["metadata"] for item in missing_batch]
            
            # Сохраняем в БД сразу
            collection.upsert(
                documents=missing_texts,
                embeddings=embeddings.tolist(),
                ids=missing_ids,
                metadatas=missing_metadatas
            )
            total_encoded += len(missing_batch)
            
        processed = min(i + batch_size, len(chunks))
        print(f"   Обраработано {processed} / {len(chunks)} фрагментов (из них {len(missing_batch)} пропущено через нейросеть).")

    print(f"   ✅ Все векторы вычислены и сохранены! Всего записей в БД: {collection.count()}")
    print(f"   📊 За этот запуск было векторизировано новых фрагментов: {total_encoded}.")



def _tokenize(text):
    """Simple tokenizer for Russian/English text."""
    return re.findall(r'[а-яёa-z0-9]+', text.lower())


def build_bm25_index(chunks):
    """Build BM25 index from chunks and save to pickle."""
    print(f"\n📚 Построение BM25 индекса из {len(chunks)} фрагментов...")

    corpus_texts = [item["text"] for item in chunks]
    corpus_ids = [item["id"] for item in chunks]
    corpus_metadatas = [item["metadata"] for item in chunks]

    tokenized_corpus = [_tokenize(text) for text in corpus_texts]

    bm25 = BM25Okapi(tokenized_corpus)

    bm25_data = {
        "bm25": bm25,
        "corpus_texts": corpus_texts,
        "corpus_ids": corpus_ids,
        "corpus_metadatas": corpus_metadatas,
    }

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_data, f)

    file_size = os.path.getsize(BM25_INDEX_PATH) / 1024 / 1024
    print(f"   ✅ BM25 индекс сохранён: {BM25_INDEX_PATH} ({file_size:.1f} MB)")


if __name__ == "__main__":
    all_chunks = []
    for pdf_path in PDF_PATHS:
        if not os.path.exists(pdf_path):
            print(f"⚠️ Файл не найден: {pdf_path}")
            continue

        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_text_with_context(pages, CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("❌ Не найдено фрагментов для добавления.")
        exit(1)

    print(f"\n📊 Итого: {len(all_chunks)} фрагментов из {len(PDF_PATHS)} PDF-файлов.")

    # Мы больше не сносим старую БД при каждом запуске! Скрипт просто дозапишет недостающие.

    print(f"\n🔧 Загрузка модели {EMBED_MODEL_NAME}...")
    model = SentenceTransformer(EMBED_MODEL_NAME, trust_remote_code=True)
    print(f"   ✅ Модель загружена.")

    # Вычисляем и сохраняем батчами
    process_and_save_embeddings(model, all_chunks)

    print("\n🎉 Готово!")
