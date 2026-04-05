import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

RAW_DATA_DIR = "data/raw"
CHROMA_DIR = "chroma_db"

USE_LARGE_MODEL = os.getenv("USE_LARGE_MODEL", "False") == "True"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5" if USE_LARGE_MODEL else "all-MiniLM-L6-v2"


def load_documents() -> list[dict]:
    documents = []
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith(".txt"):
            company_name = filename.replace("_10k.txt", "")
            filepath = os.path.join(RAW_DATA_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append({"text": text, "company": company_name})
            print(f"  Loaded {company_name} ({len(text):,} characters)")
    return documents


def chunk_documents(documents: list[dict]) -> tuple[list, list]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    all_chunks = []
    all_metadatas = []

    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append(
                {
                    "company": doc["company"],
                    "chunk_index": i,
                    "source": f"{doc['company']}_10k",
                }
            )
        print(f"  {doc['company']}: {len(chunks)} chunks")

    return all_chunks, all_metadatas


def ingest() -> None:
    print("=== Starting Ingestion ===\n")
    print(f"Embedding model: {EMBEDDING_MODEL}\n")

    print("Step 1: Loading documents...")
    documents = load_documents()

    print("\nStep 2: Chunking documents...")
    chunks, metadatas = chunk_documents(documents)
    print(f"\nTotal chunks: {len(chunks)}")

    print("\nStep 3: Loading embedding model...")
    print("(First time will download the model — wait for it)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
    )
    print("Embedding model ready")

    print("\nStep 4: Building vector store...")
    Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_DIR,
        collection_name="sec_filings",
    )

    print(f"\n=== Ingestion Complete ===")  # noqa: F541
    print(f"Collection: sec_filings")  # noqa: F541
    print(f"Total chunks stored: {len(chunks)}")
    print(f"ChromaDB saved to: {CHROMA_DIR}/")


if __name__ == "__main__":
    ingest()
