import os
import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq, APIStatusError, InternalServerError, RateLimitError
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
TOP_K = 8
MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "5"))
BASE_RETRY_DELAY = int(os.getenv("GROQ_BASE_RETRY_DELAY", "15"))


def load_vectorstore() -> Chroma:
    """
    Load the existing ChromaDB vector store from disk
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="sec_filings",
    )

    return vectorstore


def _sleep_for_retry(attempt: int, error: Exception) -> None:
    """
    Sleep with exponential backoff when Groq asks us to slow down.
    """
    retry_after = None
    response = getattr(error, "response", None)

    if response is not None:
        headers = getattr(response, "headers", None) or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")

    try:
        delay = int(float(retry_after)) if retry_after else BASE_RETRY_DELAY * attempt
    except (TypeError, ValueError):
        delay = BASE_RETRY_DELAY * attempt

    print(f"  Groq limit hit. Waiting {delay}s before retry {attempt}/{MAX_RETRIES}...")
    time.sleep(delay)


def _create_chat_completion(messages: list, model: str, temperature: float) -> str:
    """
    Wrap Groq calls with retries for rate limits and transient API errors.
    """
    client = Groq(api_key=GROQ_API_KEY)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except (RateLimitError, InternalServerError) as error:
            if attempt == MAX_RETRIES:
                raise
            _sleep_for_retry(attempt, error)
        except APIStatusError as error:
            if (
                error.status_code not in {429, 500, 502, 503, 504}
                or attempt == MAX_RETRIES
            ):
                raise
            _sleep_for_retry(attempt, error)


def _stream_chat_completion(messages: list, model: str, temperature: float):
    """
    Stream a Groq completion token-by-token with the same retry logic used by
    normal completions.
    """
    client = Groq(api_key=GROQ_API_KEY)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta
            return
        except (RateLimitError, InternalServerError) as error:
            if attempt == MAX_RETRIES:
                raise
            _sleep_for_retry(attempt, error)
        except APIStatusError as error:
            if (
                error.status_code not in {429, 500, 502, 503, 504}
                or attempt == MAX_RETRIES
            ):
                raise
            _sleep_for_retry(attempt, error)


def rewrite_query(query: str) -> str:
    """
    Rewrite query to be more specific for better retrieval.
    """
    return _create_chat_completion(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": f"""Rewrite this question to be more specific for searching SEC 10-K filings.
Focus on key financial terms and concepts.
Return ONLY the rewritten question, nothing else.

Original question: {query}
Rewritten question:""",
            }
        ],
        temperature=0.0,
    )


def retrieve_chunks(
    vectorstore: Chroma, query: str, company_filter: str = None
) -> list:
    """
    Search ChromaDB for the most relevant chunks.
    Optionally filter by company.
    """
    if company_filter:
        results = vectorstore.similarity_search(
            query, k=TOP_K, filter={"company": company_filter}
        )
    else:
        results = vectorstore.similarity_search(query, k=TOP_K)

    return results


def rerank_chunks(query: str, chunks: list, top_n: int = 4) -> list:
    """
    Use LLM to rerank chunks by relevance to the query.
    Returns top_n most relevant chunks.
    """
    if len(chunks) <= top_n:
        return chunks

    # Build a numbered list of chunks for the LLM to rank
    chunks_text = ""
    for i, chunk in enumerate(chunks):
        chunks_text += f"\n[{i}] {chunk.page_content[:300]}\n"

    raw = _create_chat_completion(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": f"""You are a relevance ranking system for SEC 10-K filings.

Given a question and a list of text chunks, rank the chunks by how relevant they are to answering the question.

Question: {query}

Chunks:
{chunks_text}

Return ONLY a JSON array of the chunk indices in order from most to least relevant.
Example: [3, 0, 5, 2, 1, 4, 6, 7]
Return ONLY the JSON array, nothing else.""",
            }
        ],
        temperature=0.0,
    )

    try:
        # Parse the ranked indices
        import json

        ranked_indices = json.loads(raw)
        # Return top_n chunks in ranked order
        reranked = [chunks[i] for i in ranked_indices[:top_n] if i < len(chunks)]
        print(f"  Reranked chunks: {ranked_indices[:top_n]}")
        return reranked
    except Exception as e:
        print(f"  Reranking failed: {e} — using original order")
        return chunks[:top_n]


def build_prompt(query: str, chunks: list) -> str:
    """
    Build the prompt we send to Groq.
    Combines retrieved chunks + user question.
    """
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

    # Step 2: Rewrite query for better retrieval
    rewritten_query = rewrite_query(query)
    print(f"  Rewritten query: {rewritten_query}")

    # Step 3: Find relevant chunks using rewritten query
    chunks = retrieve_chunks(vectorstore, rewritten_query, company_filter)

    # Step 4: Rerank chunks by relevance
    chunks = rerank_chunks(query, chunks, top_n=4)

    # Step 5: Build prompt using original query
    prompt = build_prompt(query, chunks)

    # Step 6: Call Groq API
    answer = _create_chat_completion(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    # Step 7: Format sources
    sources = [
        {
            "company": chunk.metadata.get("company"),
            "chunk_index": chunk.metadata.get("chunk_index"),
            "text_preview": chunk.page_content[:150] + "...",
        }
        for chunk in chunks
    ]

    return {"question": query, "answer": answer, "sources": sources}


def get_answer_stream(query: str, company_filter: str = None, vectorstore=None):
    """
    Run the same RAG pipeline as get_answer(), but stream the final answer text
    as it is generated.
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()

    rewritten_query = rewrite_query(query)
    print(f"  Rewritten query: {rewritten_query}")

    chunks = retrieve_chunks(vectorstore, rewritten_query, company_filter)
    chunks = rerank_chunks(query, chunks, top_n=4)
    prompt = build_prompt(query, chunks)

    sources = [
        {
            "company": chunk.metadata.get("company"),
            "chunk_index": chunk.metadata.get("chunk_index"),
            "text_preview": chunk.page_content[:150] + "...",
        }
        for chunk in chunks
    ]

    answer = ""
    for token in _stream_chat_completion(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    ):
        answer += token
        yield {"question": query, "answer": answer, "sources": sources}


if __name__ == "__main__":
    print("=== Testing Retriever ===\n")

    # result = get_answer("What export control risks does NVIDIA face?")
    # result = get_answer("Which companies mention AI as a growth opportunity?")
    result = get_answer(
        "How does Amazon describe its AWS business?", company_filter="amazon"
    )

    print(f"Question: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources used:")  # noqa: F541
    for s in result["sources"]:
        print(f"  - {s['company']} | chunk {s['chunk_index']}")
        print(f"    Preview: {s['text_preview']}")
        print()
