import json
import os
import sys

import gradio as gr
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

CHROMA_DIR = "chroma_db"
COMPANIES = ["All Companies", "nvidia", "apple", "microsoft", "tesla", "amazon"]
QUICK_QUESTIONS = [
    {
        "label": "NVIDIA export risks",
        "question": "What export control risks does NVIDIA face?",
        "company": "nvidia",
    },
    {
        "label": "Apple revenue sources",
        "question": "What were Apple's main revenue sources?",
        "company": "apple",
    },
    {
        "label": "Microsoft AI investments",
        "question": "What are Microsoft's key AI investments?",
        "company": "microsoft",
    },
    {
        "label": "Tesla autonomy risks",
        "question": "What risks does Tesla mention regarding autonomous driving?",
        "company": "tesla",
    },
    {
        "label": "Amazon AWS business",
        "question": "How does Amazon describe its AWS business?",
        "company": "amazon",
    },
    {
        "label": "Competition overview",
        "question": "What do these companies say about competition?",
        "company": "All Companies",
    },
]

DEFAULT_ANSWER = """### Answer

Ask a question about one company or all companies to see a grounded response here.
"""

DEFAULT_SOURCES = """### Sources

Relevant filing excerpts will appear here after a response is generated.
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Manrope:wght@400;500;600;700&display=swap');

:root {
    --bg: #1f1f1b;
    --bg-soft: #272723;
    --panel: rgba(48, 48, 43, 0.92);
    --panel-strong: rgba(56, 56, 50, 0.96);
    --border: rgba(255, 255, 255, 0.08);
    --border-strong: rgba(255, 255, 255, 0.14);
    --text: #ede8de;
    --muted: #b8b1a4;
    --subtle: #918978;
    --accent: #d88a67;
    --accent-soft: rgba(216, 138, 103, 0.16);
    --shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
}

* {
    box-sizing: border-box;
}

html, body, .gradio-container {
    min-height: 100%;
    background:
        radial-gradient(circle at top, rgba(216, 138, 103, 0.09), transparent 30%),
        linear-gradient(180deg, #262520 0%, #1f1f1b 46%, #1b1b18 100%) !important;
    color: var(--text) !important;
    font-family: 'Manrope', sans-serif !important;
}

.gradio-container {
    max-width: 1220px !important;
    margin: 0 auto !important;
    padding: 24px 24px 40px !important;
}

.app-frame {
    max-width: 1060px;
    margin: 0 auto;
}

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 24px;
}

.brand-lockup {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-mark {
    width: 14px;
    height: 14px;
    border-radius: 999px;
    background: var(--accent);
    box-shadow: 0 0 0 6px rgba(216, 138, 103, 0.12);
}

.brand-title {
    font-family: 'Fraunces', serif;
    font-size: 1.4rem;
    line-height: 1;
    color: var(--text);
}

.brand-subtitle {
    color: var(--subtle);
    font-size: 0.92rem;
}

.topbar-note {
    padding: 10px 14px;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: rgba(17, 17, 15, 0.45);
    color: var(--muted);
    font-size: 0.92rem;
}

.app-tabs {
    margin-top: 8px;
}

.tab-nav {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    margin-bottom: 16px !important;
}

.tab-nav button {
    background: transparent !important;
    color: var(--subtle) !important;
    border: none !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.8rem 1rem !important;
}

.tab-nav button.selected {
    color: var(--text) !important;
    border-bottom: 2px solid var(--accent) !important;
}

.chat-stage, .eval-stage {
    max-width: 980px;
    margin: 0 auto;
}

.hero-panel {
    text-align: center;
    padding: 44px 0 28px;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 14px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(14, 14, 12, 0.42);
    color: var(--muted);
    font-size: 0.92rem;
    margin-bottom: 28px;
}

.hero-badge::before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: var(--accent);
}

.hero-title {
    font-family: 'Fraunces', serif;
    font-size: clamp(2.8rem, 5vw, 4.4rem);
    line-height: 1.02;
    margin: 0;
    color: var(--text);
}

.hero-subtitle {
    max-width: 640px;
    margin: 14px auto 0;
    color: var(--muted);
    font-size: 1.03rem;
    line-height: 1.65;
}

.composer-shell {
    max-width: 860px;
    margin: 0 auto;
    padding: 12px;
    border-radius: 28px;
    background: var(--panel);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}

.composer-shell textarea {
    min-height: 150px !important;
    resize: none !important;
    background: transparent !important;
    border: none !important;
    color: var(--text) !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.12rem !important;
    line-height: 1.6 !important;
    box-shadow: none !important;
    padding: 18px 20px 10px !important;
}

.composer-shell textarea::placeholder {
    color: var(--subtle) !important;
}

.composer-controls {
    align-items: center;
    gap: 12px;
    padding: 0 10px 8px;
}

.company-select {
    flex: 1 1 280px !important;
}

.company-select .wrap,
.company-select select {
    background: rgba(22, 22, 19, 0.72) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    color: var(--text) !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.96rem !important;
    min-height: 46px !important;
}

.ask-button, .eval-button {
    flex: 0 0 auto !important;
}

.ask-button button,
.eval-button button {
    min-height: 46px !important;
    border-radius: 999px !important;
    border: none !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.96rem !important;
    font-weight: 700 !important;
    padding: 0 22px !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}

.ask-button button {
    background: var(--accent) !important;
    color: #181713 !important;
    box-shadow: 0 14px 30px rgba(216, 138, 103, 0.22) !important;
}

.eval-button button {
    background: transparent !important;
    color: var(--text) !important;
    border: 1px solid var(--border-strong) !important;
}

.ask-button button:hover,
.eval-button button:hover,
.quick-chip button:hover {
    transform: translateY(-1px) !important;
}

.quick-section-label {
    margin: 18px auto 12px;
    max-width: 860px;
    color: var(--subtle);
    font-size: 0.92rem;
}

.quick-chip-row {
    max-width: 860px;
    margin: 0 auto 24px;
    gap: 10px;
    flex-wrap: wrap !important;
}

.quick-chip {
    min-width: 0 !important;
}

.quick-chip button {
    border-radius: 999px !important;
    border: 1px solid var(--border) !important;
    background: rgba(36, 36, 32, 0.78) !important;
    color: var(--text) !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.7rem 1rem !important;
}

.results-row {
    margin-top: 18px;
    gap: 18px;
}

.result-panel {
    border-radius: 24px;
    border: 1px solid var(--border);
    background: var(--panel-strong);
    padding: 8px;
    min-height: 320px;
}

.result-panel .prose,
.result-panel .message-wrap,
.result-panel .md {
    color: var(--text) !important;
}

.result-panel h1,
.result-panel h2,
.result-panel h3,
.result-panel strong {
    color: var(--text) !important;
}

.result-panel p,
.result-panel li,
.result-panel code {
    color: var(--muted) !important;
}

.result-panel hr {
    border-color: var(--border) !important;
}

.eval-hero {
    padding: 14px 0 8px;
}

.eval-title {
    font-family: 'Fraunces', serif;
    font-size: 2.2rem;
    margin: 0 0 10px;
    color: var(--text);
}

.eval-copy {
    max-width: 700px;
    color: var(--muted);
    line-height: 1.7;
    margin: 0 0 18px;
}

.eval-summary {
    border-radius: 22px;
    border: 1px solid var(--border);
    background: var(--panel-strong);
    padding: 8px;
}

.dataframe {
    border-radius: 22px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    background: var(--panel-strong) !important;
}

label, .label-wrap {
    color: var(--subtle) !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.9rem !important;
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 18px 16px 28px !important;
    }

    .topbar {
        flex-direction: column;
        align-items: flex-start;
    }

    .hero-panel {
        padding-top: 28px;
    }

    .composer-shell textarea {
        min-height: 132px !important;
        font-size: 1rem !important;
    }

    .results-row {
        flex-direction: column !important;
    }
}
"""

HEADER_HTML = """
<div class="app-frame">
    <div class="topbar">
        <div class="brand-lockup">
            <div class="brand-mark"></div>
            <div>
                <div class="brand-title">SEC Filing RAG</div>
                <div class="brand-subtitle">Grounded analysis for NVIDIA, Apple, Microsoft, Tesla, and Amazon</div>
            </div>
        </div>
        <div class="topbar-note">Saved eval mode keeps Groq usage under control</div>
    </div>
</div>
"""

ASK_HERO_HTML = """
<div class="hero-panel">
    <div class="hero-badge">10-K research assistant</div>
    <h1 class="hero-title">What do you want to learn from the filings?</h1>
    <p class="hero-subtitle">
        Ask about risks, revenue, AI strategy, AWS, competition, or company-specific details. Pick a company in the composer or use a quick question to test the system instantly.
    </p>
</div>
"""

EVAL_INTRO_HTML = """
<div class="eval-hero">
    <h2 class="eval-title">Evaluation Dashboard</h2>
    <p class="eval-copy">
        This dashboard reads the saved <code>eval_results.json</code> file from disk instead of re-running the judge pipeline. That keeps the UI responsive and avoids burning through the free-tier Groq limit.
    </p>
</div>
"""


def run_startup_ingestion():
    """
    Only runs when chroma_db does not exist.
    Fetches SEC filings and builds the vector store from scratch.
    """
    print("=== No ChromaDB found. Running startup ingestion... ===")

    print("Step 1: Fetching SEC filings...")
    from rag.fetcher import fetch_all

    fetch_all()
    print("Fetching complete.")

    print("Step 2: Building vector store...")
    from rag.ingestor import ingest

    ingest()
    print("Ingestion complete.")


if not os.path.exists(CHROMA_DIR):
    run_startup_ingestion()
else:
    print("ChromaDB found. Skipping ingestion.")

from rag.retriever import get_answer, load_vectorstore  # noqa: E402

print("Loading vector store...")
vectorstore = load_vectorstore()
print("Ready.")


def format_sources(sources: list) -> str:
    if not sources:
        return "### Sources\n\nNo sources found."

    lines = ["### Sources", ""]
    for source in sources:
        company = source.get("company", "unknown").upper()
        chunk = source.get("chunk_index", "?")
        preview = source.get("text_preview", "")
        lines.append(f"**{company} - chunk {chunk}**")
        lines.append(preview)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines[:-2])


def ask_question(question: str, company: str) -> tuple[str, str]:
    question = question.strip()
    if not question:
        return "### Answer\n\nPlease enter a question.", DEFAULT_SOURCES

    company_filter = None if company == "All Companies" else company

    try:
        result = get_answer(question, company_filter, vectorstore)
        answer = f"### Answer\n\n{result['answer']}"
        sources = format_sources(result["sources"])
        return answer, sources
    except Exception as error:
        return f"### Answer\n\nError: {str(error)}", DEFAULT_SOURCES


def run_quick_question(question: str, company: str) -> tuple[str, str, str, str]:
    answer, sources = ask_question(question, company)
    return question, company, answer, sources


def load_eval_results() -> tuple[pd.DataFrame, str]:
    try:
        with open("eval_results.json", "r", encoding="utf-8") as file:
            data = json.load(file)

        results = data["results"]
        summary = data["summary"]

        rows = []
        for result in results:
            question_text = result["question"]
            if len(question_text) > 60:
                question_text = question_text[:57] + "..."

            rows.append(
                {
                    "Question": question_text,
                    "Company": result["company_filter"] or "All",
                    "Faithfulness": f"{result['faithfulness']}/5",
                    "Relevance": f"{result['answer_relevance']}/5",
                    "Precision": f"{result['context_precision']}/5",
                    "Overall": f"{result['overall']}/5",
                }
            )

        dataframe = pd.DataFrame(rows)

        summary_md = f"""### Saved Evaluation Summary

| Metric | Average Score |
| --- | --- |
| Faithfulness | {summary["avg_faithfulness"]} / 5 |
| Answer Relevance | {summary["avg_relevance"]} / 5 |
| Context Precision | {summary["avg_precision"]} / 5 |
| **Overall** | **{summary["avg_overall"]} / 5** |

Loaded from `eval_results.json` on disk. No new Groq evaluation calls were made.
"""
        return dataframe, summary_md

    except FileNotFoundError:
        return pd.DataFrame(), "No evaluation results found. Run `python .\\rag\\evaluator.py` first."
    except Exception as error:
        return pd.DataFrame(), f"Error loading saved evaluation results: {str(error)}"


with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="SEC Filing RAG") as app:
    gr.HTML(HEADER_HTML)

    with gr.Tabs(elem_classes=["app-tabs"]):
        with gr.Tab("Ask"):
            with gr.Column(elem_classes=["chat-stage"]):
                gr.HTML(ASK_HERO_HTML)

                with gr.Column(elem_classes=["composer-shell"]):
                    question_input = gr.Textbox(
                        placeholder="Ask about revenue, competition, export controls, AWS, AI strategy, or filing risks...",
                        show_label=False,
                        lines=4,
                        container=False,
                        elem_classes=["composer-input"],
                    )

                    with gr.Row(elem_classes=["composer-controls"]):
                        company_input = gr.Dropdown(
                            choices=COMPANIES,
                            value="All Companies",
                            show_label=False,
                            container=False,
                            elem_classes=["company-select"],
                        )
                        submit_btn = gr.Button(
                            "Ask",
                            variant="primary",
                            elem_classes=["ask-button"],
                        )

                gr.HTML('<div class="quick-section-label">Quick questions</div>')

                quick_buttons = []
                with gr.Row(elem_classes=["quick-chip-row"]):
                    for item in QUICK_QUESTIONS:
                        quick_buttons.append(
                            gr.Button(
                                item["label"],
                                variant="secondary",
                                elem_classes=["quick-chip"],
                            )
                        )

                with gr.Row(elem_classes=["results-row"]):
                    with gr.Column(elem_classes=["result-panel"]):
                        answer_output = gr.Markdown(DEFAULT_ANSWER)

                    with gr.Column(elem_classes=["result-panel"]):
                        sources_output = gr.Markdown(DEFAULT_SOURCES)

        with gr.Tab("Eval Dashboard"):
            with gr.Column(elem_classes=["eval-stage"]):
                gr.HTML(EVAL_INTRO_HTML)
                eval_btn = gr.Button(
                    "Load Saved Evaluation",
                    variant="secondary",
                    elem_classes=["eval-button"],
                )
                eval_table = gr.Dataframe(
                    label="Saved Results",
                    wrap=True,
                    interactive=False,
                )
                eval_summary = gr.Markdown(
                    "Saved evaluation metrics will appear here after loading the disk results.",
                    elem_classes=["eval-summary"],
                )

    submit_btn.click(
        fn=ask_question,
        inputs=[question_input, company_input],
        outputs=[answer_output, sources_output],
        show_progress="full",
    )

    question_input.submit(
        fn=ask_question,
        inputs=[question_input, company_input],
        outputs=[answer_output, sources_output],
        show_progress="full",
    )

    for button, item in zip(quick_buttons, QUICK_QUESTIONS):
        button.click(
            fn=lambda q=item["question"], c=item["company"]: run_quick_question(q, c),
            inputs=[],
            outputs=[question_input, company_input, answer_output, sources_output],
            show_progress="full",
        )

    eval_btn.click(
        fn=load_eval_results,
        inputs=[],
        outputs=[eval_table, eval_summary],
        show_progress="full",
    )

app.queue().launch()
