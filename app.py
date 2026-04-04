import gradio as gr
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.retriever import get_answer, load_vectorstore
from rag.evaluator import run_evaluation

# Load vectorstore once at startup
print("Loading vector store...")
vectorstore = load_vectorstore()
print("Ready.")

COMPANIES = ["All Companies", "nvidia", "apple", "microsoft", "tesla", "amazon"]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background-color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 2.5rem 0 2rem 0;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 2rem;
}

.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #f5c842;
    letter-spacing: -0.5px;
    margin: 0;
}

.app-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #4a4a6a;
    margin-top: 0.5rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Tabs */
.tab-nav {
    background: transparent !important;
    border-bottom: 1px solid #1e1e2e !important;
}

.tab-nav button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #4a4a6a !important;
    background: transparent !important;
    border: none !important;
    padding: 0.75rem 1.5rem !important;
}

.tab-nav button.selected {
    color: #f5c842 !important;
    border-bottom: 2px solid #f5c842 !important;
}

/* Inputs */
textarea, input[type="text"] {
    background: #0f0f1a !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 6px !important;
    color: #e0e0f0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.75rem !important;
    transition: border-color 0.2s !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #f5c842 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(245, 200, 66, 0.1) !important;
}

/* Dropdown */
select {
    background: #0f0f1a !important;
    border: 1px solid #1e1e2e !important;
    color: #e0e0f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 6px !important;
    padding: 0.6rem !important;
}

/* Buttons */
button.primary {
    background: #f5c842 !important;
    color: #0a0a0f !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.75rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}

button.primary:hover {
    background: #ffd966 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(245, 200, 66, 0.3) !important;
}

button.secondary {
    background: transparent !important;
    color: #f5c842 !important;
    border: 1px solid #f5c842 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.2s !important;
}

button.secondary:hover {
    background: rgba(245, 200, 66, 0.1) !important;
}

/* Output boxes */
.output-box {
    background: #0f0f1a !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 6px !important;
    color: #e0e0f0 !important;
    font-family: 'Syne', sans-serif !important;
    padding: 1rem !important;
}

/* Labels */
label, .label-wrap {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #4a4a6a !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    margin-bottom: 0.4rem !important;
}

/* Score cards */
.score-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
}

.score-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #4a4a6a;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.score-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #f5c842;
}

/* Dataframe */
.dataframe {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* Source tags */
.source-tag {
    display: inline-block;
    background: rgba(245, 200, 66, 0.1);
    border: 1px solid rgba(245, 200, 66, 0.3);
    color: #f5c842;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 8px;
    border-radius: 3px;
    margin: 2px;
    letter-spacing: 1px;
}
"""

HEADER_HTML = """
<div class="app-header">
    <div class="app-title">⬡ SEC FILING RAG</div>
    <div class="app-subtitle">AI-powered analysis of SEC 10-K filings · NVIDIA · Apple · Microsoft · Tesla · Amazon</div>
</div>
"""


def format_sources(sources: list) -> str:
    if not sources:
        return "No sources found."
    lines = []
    for s in sources:
        company = s.get("company", "unknown").upper()
        chunk = s.get("chunk_index", "?")
        preview = s.get("text_preview", "")
        lines.append(f"**[{company} — chunk {chunk}]**\n{preview}\n")
    return "\n---\n".join(lines)


def ask_question(question: str, company: str) -> tuple:
    if not question.strip():
        return "Please enter a question.", ""

    company_filter = None if company == "All Companies" else company

    try:
        result = get_answer(question, company_filter, vectorstore)
        answer = result["answer"]
        sources = format_sources(result["sources"])
        return answer, sources
    except Exception as e:
        return f"Error: {str(e)}", ""


def run_eval() -> tuple:
    try:
        results = run_evaluation()
        rows = []
        for r in results:
            rows.append({
                "Question": r["question"][:60] + "...",
                "Company": r["company_filter"] or "All",
                "Faithfulness": f"{r['faithfulness']}/5",
                "Relevance": f"{r['answer_relevance']}/5",
                "Precision": f"{r['context_precision']}/5",
                "Overall": f"{r['overall']}/5",
            })

        df = pd.DataFrame(rows)

        avg_overall = sum(r["overall"] for r in results) / len(results)
        avg_faith = sum(r["faithfulness"] for r in results) / len(results)
        avg_rel = sum(r["answer_relevance"] for r in results) / len(results)
        avg_prec = sum(r["context_precision"] for r in results) / len(results)

        summary = f"""### Evaluation Summary

| Metric | Average Score |
|--------|--------------|
| Faithfulness | {avg_faith:.2f} / 5 |
| Answer Relevance | {avg_rel:.2f} / 5 |
| Context Precision | {avg_prec:.2f} / 5 |
| **Overall** | **{avg_overall:.2f} / 5** |
"""
        return df, summary

    except Exception as e:
        return pd.DataFrame(), f"Error: {str(e)}"


# Update get_answer to accept vectorstore
def ask_wrapper(question, company):
    return ask_question(question, company)


with gr.Blocks(css=CSS, theme=gr.themes.Base()) as app:
    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # TAB 1 — ASK
        with gr.Tab("Ask"):
            with gr.Row():
                with gr.Column(scale=3):
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="e.g. What export control risks does NVIDIA face?",
                        lines=3
                    )
                with gr.Column(scale=1):
                    company_input = gr.Dropdown(
                        choices=COMPANIES,
                        value="All Companies",
                        label="Company Filter"
                    )

            submit_btn = gr.Button("Analyze →", variant="primary")

            with gr.Row():
                with gr.Column():
                    answer_output = gr.Markdown(label="Answer")
                with gr.Column():
                    sources_output = gr.Markdown(label="Sources")

            submit_btn.click(
                fn=ask_wrapper,
                inputs=[question_input, company_input],
                outputs=[answer_output, sources_output]
            )

        # TAB 2 — EVAL DASHBOARD
        with gr.Tab("Eval Dashboard"):
            gr.Markdown("""
### Automated Evaluation Pipeline
Runs all test questions through the RAG system and scores each answer using an LLM-as-judge on three metrics: **Faithfulness**, **Answer Relevance**, and **Context Precision**.
            """)

            eval_btn = gr.Button("Run Evaluation", variant="secondary")

            eval_table = gr.Dataframe(
                label="Results",
                wrap=True
            )
            eval_summary = gr.Markdown(label="Summary")

            eval_btn.click(
                fn=run_eval,
                inputs=[],
                outputs=[eval_table, eval_summary]
            )

app.launch()