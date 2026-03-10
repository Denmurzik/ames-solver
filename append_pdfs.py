import os
import chromadb
import fitz  # PyMuPDF
import hashlib
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ============================================================
# Настройки
# ============================================================
PDF_PATHS = [
    "/home/denis/ames-suck-my-ass/pdf24_merged.pdf",
    "/home/denis/ames-suck-my-ass/Irtegov-OS-Unix-System-Calls.pdf",
]
CHROMA_DB_DIR = "/home/denis/ames-suck-my-ass/chroma_db"
COLLECTION_NAME = "book_knowledge"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400

# Контекстная модель Perplexity
EMBED_MODEL_NAME = "perplexity-ai/pplx-embed-context-v1-0.6B"

# Сколько соседних страниц брать для формирования контекста
CONTEXT_WINDOW_PAGES = 1

# Максимальная длина контекста в символах
MAX_CONTEXT_CHARS = 6000

# Размер батча для encode()
ENCODE_BATCH_SIZE = 16


def extract_text_from_pdf(pdf_path):
    """Извлекает текст из каждой страницы PDF."""
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

    print(f"   ✅ [{os.path.basename(pdf_path)}] Извлечено страниц с текстом: {len(pages_data)}.")
    return pages_data


def build_page_context(pages_data, current_idx, window=CONTEXT_WINDOW_PAGES, max_chars=MAX_CONTEXT_CHARS):
    """Формирует контекст из соседних страниц."""
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
    """Разбивает текст на чанки с контекстом из соседних страниц."""
    print("🔪 Разбиение текста на чанки с контекстом...")

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
            chunks.append({
                "text": chunk,
                "context": context,
                "metadata": {"source": source_name, "page": page_num}
            })

    print(f"   ✅ Создано {len(chunks)} фрагментов с контекстом.")
    return chunks


def compute_contextual_embeddings(model, chunks, batch_size=ENCODE_BATCH_SIZE):
    """Вычисляет эмбеддинги с контекстом: [context, passage]."""
    print(f"🧠 Вычисление контекстных эмбеддингов ({len(chunks)} чанков)...")

    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        pairs = [[item["context"], item["text"]] for item in batch]
        embeddings = model.encode(pairs, normalize_embeddings=True)
        all_embeddings.append(embeddings)

        processed = min(i + batch_size, len(chunks))
        if processed % (batch_size * 5) == 0 or processed == len(chunks):
            print(f"   Векторизировано {processed} / {len(chunks)} фрагментов...")

    all_embeddings = np.vstack(all_embeddings)
    print(f"   ✅ Векторизация завершена. Размерность: {all_embeddings.shape}")
    return all_embeddings


def append_to_chromadb(chunks, embeddings):
    """Добавляет новые чанки с предвычисленными эмбеддингами в существующую ChromaDB."""
    print("💾 Подключение к ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    start_count = collection.count()
    print(f"   В базе сейчас {start_count} записей.")

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]

        ids = []
        for item in batch_chunks:
            hash_str = f"{item['metadata']['source']}_{item['metadata']['page']}_{hashlib.md5(item['text'].encode()).hexdigest()}"
            ids.append(hash_str)

        texts = [item["text"] for item in batch_chunks]
        metadatas = [item["metadata"] for item in batch_chunks]

        collection.upsert(
            documents=texts,
            embeddings=batch_embeddings.tolist(),
            ids=ids,
            metadatas=metadatas
        )

        loaded = min(i + batch_size, len(chunks))
        print(f"   Загружено {loaded} / {len(chunks)} фрагментов...")

    end_count = collection.count()
    print(f"   ✅ Готово! В базе теперь {end_count} записей (добавлено {end_count - start_count}).")
    return start_count, end_count


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

    print(f"\n🔧 Загрузка модели {EMBED_MODEL_NAME}...")
    model = SentenceTransformer(EMBED_MODEL_NAME, trust_remote_code=True)
    print(f"   ✅ Модель загружена.")

    embeddings = compute_contextual_embeddings(model, all_chunks)

    append_to_chromadb(all_chunks, embeddings)

    print("\n🎉 Готово! Новые PDF-файлы добавлены в базу знаний.")
