import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

load_dotenv()

CHROMA_DIR = "chroma_db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

USE_LARGE_MODEL = os.getenv("USE_LARGE_MODEL", "False") == "True"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5" if USE_LARGE_MODEL else "all-MiniLM-L6-v2"

# How many chunks to retrieve per query
TOP_K = 5


def load_vectorstore() -> Chroma:
    """
    Load the existing ChromaDB vector store from disk
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="sec_filings"
    )

    return vectorstore


def retrieve_chunks(vectorstore: Chroma, query: str, company_filter: str = None) -> list:
    """
    Search ChromaDB for the most relevant chunks.
    Optionally filter by company.
    """
    if company_filter:
        # Search only within a specific company
        results = vectorstore.similarity_search(
            query,
            k=TOP_K,
            filter={"company": company_filter}
        )
    else:
        # Search across all companies
        results = vectorstore.similarity_search(query, k=TOP_K)

    return results


def build_prompt(query: str, chunks: list) -> str:
    """
    Build the prompt we send to Groq.
    Combines retrieved chunks + user question.
    """
    # Join all chunk texts into one context block
    context = "\n\n---\n\n".join([chunk.page_content for chunk in chunks])

    prompt = f"""You are a financial analyst assistant. 
You answer questions based strictly on the provided context from SEC 10-K filings.

Rules:
- Use the information from the context below to answer the question
- Be specific and mention which company you are referring to
- If the context contains relevant information, use it even if it doesn't perfectly match the question wording
- Only say you cannot find information if the context is truly unrelated to the question
- Keep answers clear and concise

Context from SEC filings:
{context}

Question: {query}

Answer:"""

    return prompt


def get_answer(query: str, company_filter: str = None, vectorstore=None) -> dict:
    """
    Main function — takes a question, returns answer + source chunks
    """
    # Step 1: Load vector store
    if vectorstore is None:
        vectorstore = load_vectorstore()

    # Step 2: Find relevant chunks
    chunks = retrieve_chunks(vectorstore, query, company_filter)

    # Step 3: Build prompt
    prompt = build_prompt(query, chunks)

    # Step 4: Call Groq API
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1  # low temperature = more factual, less creative
    )

    answer = response.choices[0].message.content

    # Step 5: Format sources
    sources = [
        {
            "company": chunk.metadata.get("company"),
            "chunk_index": chunk.metadata.get("chunk_index"),
            "text_preview": chunk.page_content[:150] + "..."
        }
        for chunk in chunks
    ]

    return {
        "question": query,
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    print("=== Testing Retriever ===\n")

    # Test question
    # result = get_answer("What export control risks does NVIDIA face?")
    # result = get_answer("Which companies mention AI as a growth opportunity?")
    result = get_answer("How does Amazon describe its AWS business?", company_filter="amazon")

    print(f"Question: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources used:")
    for s in result['sources']:
        print(f"  - {s['company']} | chunk {s['chunk_index']}")
        print(f"    Preview: {s['text_preview']}")
        print()