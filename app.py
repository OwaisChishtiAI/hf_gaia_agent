import base64
import contextlib
import io
import json
import re
from pathlib import Path
from typing import Annotated, TypedDict

import gradio as gr
import pandas as pd
import requests
import wikipedia
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
DEFAULT_AGENT_CODE_URL = "https://github.com/OwaisChishtiAI/hf_gaia_agent.git"
CACHE_JSON_PATH = Path("gaia_answers_cache.json")
RESULTS_CSV_PATH = Path("gaia_results_log.csv")
DOWNLOADS_DIR = Path("gaia_task_files")
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

MAX_TOOL_ROUNDS = 5  # Prevents infinite tool loops; forces final answer after 5 rounds

SYSTEM_PROMPT = """You are a precise research and reasoning agent answering GAIA benchmark questions.

CRITICAL OUTPUT FORMAT:
- Do your thinking/tool calls first, then end your final message with a single line in this exact format:
FINAL ANSWER: <exact answer>
- Nothing should appear after `FINAL ANSWER: <exact answer>`.
- `<exact answer>` must contain ONLY the answer value—no full sentences, no explanations, no extra labels.
- Numbers: digits only (no commas like 1,000, no units/currency symbols unless explicitly requested).
- Strings/names: omit leading articles ("a", "an", "the") unless part of a proper name. Use full city names without abbreviations (e.g., "Saint Petersburg" instead of "St. Petersburg").
- Lists: comma-separated as `item1, item2, item3`.

TOOL RULES:
- Use `python_interpreter` for math, reversing text, checking Cayley/math tables, sorting, and running code.
- Use `wikipedia_retriever` or `web_search` + `fetch_webpage_text` for factual lookups. Do not repeat the exact same search query twice.
"""

_VISION_LLM = None
_WHISPER_MODEL = None


def get_vision_llm() -> ChatOllama:
    global _VISION_LLM
    if _VISION_LLM is None:
        _VISION_LLM = ChatOllama(
            model="qwen3-vl:4b",
            temperature=0,
            num_ctx=8192,
            num_predict=1024,
            reasoning=False,
            validate_model_on_init=True,
        )
    return _VISION_LLM


def get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        from faster_whisper import WhisperModel
        _WHISPER_MODEL = WhisperModel("base", device="auto", compute_type="int8")
    return _WHISPER_MODEL


def encode_image_b64(image_path: str) -> tuple[str, str]:
    path = Path(image_path)
    ext = path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return b64, mime


# ==============================================================================
# TOOLS & RETRIEVERS (Bounded outputs so context never overflows)
# ==============================================================================

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo for facts, articles, sports stats, papers, or general info."""
    try:
        results = list(DDGS().text(query, max_results=max_results))
        if not results:
            return f"No web results found for: {query}"
        formatted = []
        for idx, r in enumerate(results, 1):
            formatted.append(
                f"[{idx}] {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')[:350]}"
            )
        return "\n\n".join(formatted)[:3000]
    except Exception as e:
        return f"web_search error: {e}"


@tool
def fetch_webpage_text(url: str, max_chars: int = 3500) -> str:
    """Fetch and extract readable text from a URL (articles, papers, web pages)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n"))
        return text.strip()[:max_chars]
    except Exception as e:
        return f"fetch_webpage_text error for {url}: {e}"


@tool
def wikipedia_retriever(query: str) -> str:
    """Search Wikipedia and retrieve the summary and key text of the top matching article."""
    try:
        titles = wikipedia.search(query, results=4)
        if not titles:
            return f"No Wikipedia pages found for: {query}"
        output_blocks = [f"Search matches: {', '.join(titles)}"]
        for title in titles[:1]:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                output_blocks.append(
                    f"=== {page.title} ({page.url}) ===\n{page.content[:3200]}"
                )
            except wikipedia.DisambiguationError as de:
                output_blocks.append(f"Disambiguation for '{title}': {de.options[:8]}")
            except Exception as inner_e:
                output_blocks.append(f"Could not load '{title}': {inner_e}")
        return "\n\n".join(output_blocks)[:3600]
    except Exception as e:
        return f"wikipedia_retriever error: {e}"


@tool
def python_interpreter(code: str) -> str:
    """Execute Python code for math, string reversal, table checks, data analysis, or scripts. Use print() to output results."""
    stdout_buf = io.StringIO()
    local_env = {}
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, {"__builtins__": __builtins__}, local_env)
        out = stdout_buf.getvalue().strip()
        if out:
            return out[:3000]
        if "result" in local_env:
            return str(local_env["result"])[:3000]
        return "Executed successfully (no stdout printed; call print(...) to see values)."
    except Exception as e:
        return f"Python execution error: {e}"


ALL_TOOLS = [
    web_search,
    fetch_webpage_text,
    wikipedia_retriever,
    python_interpreter,
]


# ==============================================================================
# ATTACHMENT DOWNLOADER & PRE-PROCESSOR
# ==============================================================================

def download_task_file(api_url: str, task_id: str, file_name: str | None = None) -> Path | None:
    """Downloads any file attached to a GAIA task from /files/{task_id}."""
    if not file_name:
        return None
    safe_name = Path(file_name).name
    dest = DOWNLOADS_DIR / f"{task_id}_{safe_name}"
    if dest.exists() and dest.stat().st_size > 0:
        return dest.resolve()

    headers = {"User-Agent": "Mozilla/5.0"}
    candidate_urls = [
        f"{api_url}/files/{task_id}",
        f"{api_url}/files/{task_id}/{safe_name}",
    ]
    for url in candidate_urls:
        try:
            resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            if resp.status_code == 200 and resp.content:
                dest.write_bytes(resp.content)
                print(f"  [OK] Downloaded {safe_name} ({len(resp.content)} bytes) -> {dest}")
                return dest.resolve()
            else:
                print(f"  [WARN] {url} returned HTTP {resp.status_code}")
        except Exception as e:
            print(f"  [WARN] Failed fetching {url}: {e}")
    return None


def preprocess_attachment(file_path: Path | None, question: str) -> str:
    """Extracts text/transcript/vision/table upfront so the 4B agent never fails on file paths."""
    if not file_path or not file_path.exists():
        return ""

    ext = file_path.suffix.lower()
    abs_path = str(file_path.resolve())

    try:
        if ext in (".png", ".jpg", ".jpeg", ".webp"):
            b64, mime = encode_image_b64(abs_path)
            msg = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"Examine this image carefully and extract all details needed to answer: {question}",
                    },
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ]
            )
            vision_out = str(get_vision_llm().invoke([msg]).content).strip()
            return f"\n\n[ATTACHED IMAGE ({file_path.name}) ANALYSIS]:\n{vision_out}"

        if ext in (".mp3", ".wav", ".m4a", ".ogg", ".flac"):
            model = get_whisper_model()
            segments, _ = model.transcribe(abs_path, beam_size=5)
            transcript = " ".join(seg.text.strip() for seg in segments).strip()
            return f"\n\n[ATTACHED AUDIO ({file_path.name}) TRANSCRIPT]:\n{transcript}"

        if ext in (".xlsx", ".xls", ".csv"):
            if ext == ".csv":
                df = pd.read_csv(file_path)
                table_str = df.to_string(index=False)
            else:
                excel = pd.ExcelFile(file_path)
                sheets = []
                for s in excel.sheet_names:
                    sheets.append(f"Sheet '{s}':\n{excel.parse(s).to_string(index=False)}")
                table_str = "\n\n".join(sheets)
            return (
                f"\n\n[ATTACHED SPREADSHEET ({file_path.name}) at '{abs_path}']:\n"
                f"{table_str[:3500]}\n"
                f"(You can also load r'{abs_path}' with pandas in `python_interpreter` for exact sums.)"
            )

        # .py, .txt, .json, .md, etc.
        raw_text = file_path.read_text(encoding="utf-8", errors="replace")[:3500]
        return (
            f"\n\n[ATTACHED FILE ({file_path.name}) at '{abs_path}']:\n"
            f"```\n{raw_text}\n```\n"
            f"(If this is Python code, execute it with `python_interpreter` to get the exact output.)"
        )
    except Exception as e:
        return f"\n\n[ATTACHMENT ERROR for {file_path.name}]: {e}"


def extract_final_answer(raw: str) -> str:
    """Extracts the exact value after FINAL ANSWER: and strips markdown/quotes."""
    text = raw.strip()
    match = re.search(r"FINAL\s*ANSWER\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip().splitlines()[0].strip()
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            text = lines[-1]
    text = re.sub(r"^(final\s*answer|answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip("`*").strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text


# ==============================================================================
# AGENT GRAPH (Bounded Tool Loops + Large Context Window)
# ==============================================================================

class GAIAAgent:
    def __init__(self, text_model: str = "qwen3.5:4b", tools: list | None = None):
        self.llm = ChatOllama(
            model=text_model,
            temperature=0,
            num_ctx=16384,     # Fixes HTTP 400 context length errors!
            num_predict=2048,  # Fixes mid-sentence output cutoffs!
            reasoning=False,
            validate_model_on_init=True,
        )
        self.tools = tools or ALL_TOOLS
        self.graph = self._build_graph()
        print(f"GAIAAgent initialized: model={text_model}, num_ctx=16384, tools={len(self.tools)}")

    def _build_graph(self):
        class AgentState(TypedDict):
            messages: Annotated[list[AnyMessage], add_messages]
            tool_rounds: int

        model_with_tools = self.llm.bind_tools(self.tools)

        def assistant(state: AgentState):
            msgs = list(state["messages"])
            if not msgs or getattr(msgs[0], "type", None) != "system":
                msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs

            rounds = state.get("tool_rounds", 0)
            if rounds >= MAX_TOOL_ROUNDS:
                # Force a final answer without tools instead of hitting recursion limit
                msgs.append(
                    HumanMessage(
                        content="Stop calling tools now. Based on what you have found, output ONLY `FINAL ANSWER: <your answer>`."
                    )
                )
                return {"messages": [self.llm.invoke(msgs)], "tool_rounds": rounds}

            response = model_with_tools.invoke(msgs)
            next_rounds = rounds + 1 if getattr(response, "tool_calls", None) else rounds
            return {"messages": [response], "tool_rounds": next_rounds}

        def route_after_assistant(state: AgentState):
            last_msg = state["messages"][-1]
            if getattr(last_msg, "tool_calls", None) and state.get("tool_rounds", 0) <= MAX_TOOL_ROUNDS:
                return "tools"
            return END

        builder = StateGraph(AgentState)
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", route_after_assistant, {"tools": "tools", END: END})
        builder.add_edge("tools", "assistant")
        return builder.compile()

    def __call__(self, question: str, file_path: Path | None = None) -> str:
        attachment_context = preprocess_attachment(file_path, question)
        full_prompt = f"{question}{attachment_context}"

        result = self.graph.invoke(
            {"messages": [HumanMessage(content=full_prompt)], "tool_rounds": 0},
            config={"recursion_limit": 25},
        )
        final_content = result["messages"][-1].content
        if isinstance(final_content, list):
            final_text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in final_content
            )
        else:
            final_text = str(final_content)
        return extract_final_answer(final_text)


# ==============================================================================
# CACHE, EVALUATION RUNNER & SUBMISSION
# ==============================================================================

def load_cache() -> dict:
    if CACHE_JSON_PATH.exists():
        try:
            return json.loads(CACHE_JSON_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache_data: dict) -> None:
    CACHE_JSON_PATH.write_text(json.dumps(cache_data, indent=2, ensure_ascii=False), encoding="utf-8")


def cache_to_df(cache_data: dict) -> pd.DataFrame:
    rows = [
        {
            "Task ID": tid,
            "Question": entry.get("question", ""),
            "File": entry.get("file_name", ""),
            "Submitted Answer": entry.get("submitted_answer", ""),
        }
        for tid, entry in cache_data.items()
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8")
    return df


def run_and_save_answers(api_url: str, overwrite_cached: bool = False):
    api_url = (api_url or DEFAULT_API_URL).strip().rstrip("/")
    questions_url = f"{api_url}/questions"

    try:
        agent = GAIAAgent(text_model="qwen3.5:4b", tools=ALL_TOOLS)
    except Exception as e:
        return f"Error initializing agent: {e}", cache_to_df(load_cache())

    try:
        resp = requests.get(questions_url, timeout=30)
        resp.raise_for_status()
        questions_data = resp.json()
        if not questions_data:
            return "Fetched questions list is empty.", cache_to_df(load_cache())
    except Exception as e:
        return f"Error fetching questions: {e}", cache_to_df(load_cache())

    cache = load_cache()
    answered_count = 0

    for idx, item in enumerate(questions_data, start=1):
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name") or item.get("filename")
        if not task_id or question_text is None:
            continue

        existing = cache.get(task_id, {}).get("submitted_answer", "")
        is_valid_cached = (
            bool(existing)
            and not existing.startswith("AGENT ERROR")
            and "cannot find the audio file" not in existing.lower()
            and "cannot see the chess" not in existing.lower()
        )
        if not overwrite_cached and is_valid_cached:
            print(f"[{idx}/{len(questions_data)}] Keeping good cached {task_id} -> {existing}")
            answered_count += 1
            continue

        local_file = download_task_file(api_url, task_id, file_name)
        print(f"[{idx}/{len(questions_data)}] Running {task_id} (file={file_name}, downloaded={bool(local_file)})...")
        try:
            ans = agent(question_text, file_path=local_file)
        except Exception as e:
            ans = f"AGENT ERROR: {e}"

        cache[task_id] = {
            "task_id": task_id,
            "question": question_text,
            "file_name": file_name or "",
            "submitted_answer": ans,
        }
        save_cache(cache)
        answered_count += 1
        print(f"[{idx}/{len(questions_data)}] Saved {task_id} -> {ans}")

    df = cache_to_df(cache)
    status = (
        f"Finished {answered_count}/{len(questions_data)} questions.\n"
        f"Saved JSON: {CACHE_JSON_PATH.resolve()}\n"
        f"Saved CSV: {RESULTS_CSV_PATH.resolve()}"
    )
    return status, df


def submit_cached_answers(username: str, agent_code: str, api_url: str):
    username = (username or "").strip()
    agent_code = (agent_code or "").strip()
    api_url = (api_url or DEFAULT_API_URL).strip().rstrip("/")
    submit_url = f"{api_url}/submit"

    if not username:
        return "Enter your exact Hugging Face username first.", cache_to_df(load_cache())
    if not agent_code:
        return "Enter your public GitHub/HF repository URL first.", cache_to_df(load_cache())

    cache = load_cache()
    answers_payload = [
        {"task_id": tid, "submitted_answer": str(entry.get("submitted_answer", "")).strip()}
        for tid, entry in cache.items()
        if entry.get("submitted_answer") is not None
    ]

    if not answers_payload:
        return f"No saved answers in {CACHE_JSON_PATH}. Run the agent first.", pd.DataFrame()

    submission_data = {
        "username": username,
        "agent_code": agent_code,
        "answers": answers_payload,
    }

    try:
        resp = requests.post(submit_url, json=submission_data, timeout=120)
        resp.raise_for_status()
        result_data = resp.json()
        return (
            f"Submission Successful!\n"
            f"User: {result_data.get('username', username)}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', '')}",
            cache_to_df(cache),
        )
    except requests.exceptions.HTTPError as e:
        detail = e.response.text[:500]
        try:
            detail = e.response.json().get("detail", detail)
        except Exception:
            pass
        return f"Submission Failed (HTTP {e.response.status_code}): {detail}", cache_to_df(cache)
    except Exception as e:
        return f"Submission Failed: {e}", cache_to_df(cache)


# ==============================================================================
# GRADIO UI
# ==============================================================================

with gr.Blocks() as demo:
    gr.Markdown("# Local Ollama GAIA Agent (Fixed Context, Bounded Tools & Auto-Attachments)")
    gr.Markdown(
        "1. Leave **Re-run ALL tasks...** unchecked to keep already-solved tasks (`b, e`, `Wojciech`, etc.) and automatically retry only the failed/errored/attachment ones.\n"
        "2. Inspect `gaia_answers_cache.json` after it finishes.\n"
        "3. Push `app.py` to your public GitHub repo, enter your **Hugging Face Username** + **Public Agent Code URL**, and click **2. Submit Saved Answers**."
    )

    with gr.Row():
        hf_username_input = gr.Textbox(label="Hugging Face Username", placeholder="your_hf_username")
        agent_code_input = gr.Textbox(label="Public Agent Code URL", value=DEFAULT_AGENT_CODE_URL)
        api_url_input = gr.Textbox(label="Scoring API URL", value=DEFAULT_API_URL)

    overwrite_checkbox = gr.Checkbox(label="Re-run ALL tasks (unchecked = retry only failed/errored tasks)", value=False)

    with gr.Row():
        run_btn = gr.Button("1. Run Agent & Save Locally", variant="primary")
        submit_btn = gr.Button("2. Submit Saved Answers")

    status_output = gr.Textbox(label="Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Saved Answers", wrap=True)

    run_btn.click(
        fn=run_and_save_answers,
        inputs=[api_url_input, overwrite_checkbox],
        outputs=[status_output, results_table],
    )
    submit_btn.click(
        fn=submit_cached_answers,
        inputs=[hf_username_input, agent_code_input, api_url_input],
        outputs=[status_output, results_table],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False)