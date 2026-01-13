import os
import json
import base64
from datetime import datetime
from io import BytesIO

import streamlit as st
import yaml
import pandas as pd
import altair as alt
from pypdf import PdfReader

try:
    from docx import Document  # provided by python-docx
except ImportError:
    Document = None

# Optional PDF-generation library
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
except ImportError:
    canvas = None
    letter = None

# External LLM clients
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import httpx


# =========================
# Constants & configuration
# =========================

ALL_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-5-mini",                 # NEW
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",       # NEW (for deep 510(k) reasoning/chat)
    "claude-3-5-sonnet-2024-10",
    "claude-3-5-haiku-20241022",
    "grok-4-fast-reasoning",
    "grok-3-mini",
]

# Models treated as OpenAI in call_llm
OPENAI_MODELS = {"gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini"}
GEMINI_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
}
ANTHROPIC_MODELS = {
    "claude-3-5-sonnet-20210",
    "claude-3-5-sonnet-2024-10",
    "claude-3-5-haiku-20241022",
}
GROK_MODELS = {"grok-4-fast-reasoning", "grok-3-mini"}

# Allowed models for agent configuration
AGENT_MODEL_CHOICES = [
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gpt-4o-mini",
    "gpt-5-mini",
]

PAINTER_STYLES = [
    "Van Gogh", "Monet", "Picasso", "Da Vinci", "Rembrandt",
    "Matisse", "Kandinsky", "Hokusai", "Yayoi Kusama", "Frida Kahlo",
    "Salvador Dali", "Rothko", "Pollock", "Chagall", "Basquiat",
    "Haring", "Georgia O'Keeffe", "Turner", "Seurat", "Escher"
]

# Localized labels for tabs
LABELS = {
    "Dashboard": {"English": "Dashboard", "繁體中文": "儀表板"},
    "510k_tab": {"English": "510(k) Intelligence", "繁體中文": "510(k) 智能分析"},
    "510k_summary_studio": {
        "English": "510(k) Summary Studio",
        "繁體中文": "510(k) 摘要視覺儀表板",
    },
    "PDF → Markdown": {"English": "PDF → Markdown", "繁體中文": "PDF → Markdown"},
    "Summary & Entities": {"English": "Summary & Entities", "繁體中文": "綜合摘要與實體"},
    "Comparator": {"English": "Comparator", "繁體中文": "文件版本比較"},
    "Checklist & Report": {"English": "Checklist & Report", "繁體中文": "審查清單與報告"},
    "Note Keeper & Magics": {"English": "Note Keeper & Magics", "繁體中文": "筆記助手與魔法"},
    "FDA Orchestration": {"English": "FDA Reviewer Orchestration", "繁體中文": "FDA 審查協同規劃"},
    "Dynamic Agents": {"English": "Dynamic Agents from Guidance", "繁體中文": "依據指引動態產生代理"},
    "Agents Config": {"English": "Agents Config Studio", "繁體中文": "代理設定工作室"},
}

# Painter style CSS snippets (simple examples)
STYLE_CSS = {
    "Van Gogh": """
      body { background: radial-gradient(circle at top left, #243B55, #141E30); }
    """,
    "Monet": """
      body { background: linear-gradient(120deg, #a1c4fd, #c2e9fb); }
    """,
    "Picasso": """
      body { background: linear-gradient(135deg, #ff9a9e, #fecfef); }
    """,
    "Da Vinci": """
      body { background: radial-gradient(circle, #f9f1c6, #c9a66b); }
    """,
    "Rembrandt": """
      body { background: radial-gradient(circle, #2c1810, #0b090a); }
    """,
    "Matisse": """
      body { background: linear-gradient(135deg, #ffecd2, #fcb69f); }
    """,
    "Kandinsky": """
      body { background: linear-gradient(135deg, #00c6ff, #0072ff); }
    """,
    "Hokusai": """
      body { background: linear-gradient(135deg, #2b5876, #4e4376); }
    """,
    "Yayoi Kusama": """
      body { background: radial-gradient(circle, #ffdd00, #ff6a00); }
    """,
    "Frida Kahlo": """
      body { background: linear-gradient(135deg, #f8b195, #f67280, #c06c84); }
    """,
    "Salvador Dali": """
      body { background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d); }
    """,
    "Rothko": """
      body { background: linear-gradient(135deg, #141E30, #243B55); }
    """,
    "Pollock": """
      body { background: repeating-linear-gradient(
        45deg,
        #222,
        #222 10px,
        #333 10px,
        #333 20px
      ); }
    """,
    "Chagall": """
      body { background: linear-gradient(135deg, #a18cd1, #fbc2eb); }
    """,
    "Basquiat": """
      body { background: linear-gradient(135deg, #f7971e, #ffd200); }
    """,
    "Haring": """
      body { background: linear-gradient(135deg, #ff512f, #dd2476); }
    """,
    "Georgia O'Keeffe": """
      body { background: linear-gradient(135deg, #ffefba, #ffffff); }
    """,
    "Turner": """
      body { background: linear-gradient(135deg, #f8ffae, #43c6ac); }
    """,
    "Seurat": """
      body { background: radial-gradient(circle, #e0eafc, #cfdef3); }
    """,
    "Escher": """
      body { background: linear-gradient(135deg, #232526, #414345); }
    """,
}


# =========================
# FDA Orchestrator Prompts
# =========================

FDA_ORCH_SYSTEM_PROMPT = """
You are an expert FDA review orchestrator with comprehensive knowledge of medical 
device regulatory review processes. Your role is to analyze device information 
and intelligently recommend which specialized review agents should be used, in 
what sequence, and with what priority.

Use the agent catalog (agents.yaml content) provided to you as the universe of available
agents. Map device characteristics to these agents, following this framework:

[THE FULL SPEC FROM USER – STEPS 1–4, TABLE FORMATS, AGENT CATEGORIES, 
MANDATORY CORE AGENTS, CONDITIONAL AGENTS, DEVICE-SPECIFIC SPECIALISTS,
PHASES 1–4, OUTPUT FORMAT, TIMELINE, FOCUS AREAS, CHALLENGES, AGENT COMMANDS, etc.]

Be thorough, specific, and provide clear rationale for each agent recommendation.
Tailor your response to the specific device characteristics provided.
If information is incomplete, note what additional details would refine the recommendation.
"""

FDA_ORCH_USER_TEMPLATE = """
Please analyze the following medical device information and provide a comprehensive 
review orchestration plan with recommended specialized agents:

**DEVICE INFORMATION:**

```
{device_information}
```

**ADDITIONAL CONTEXT** (if provided):
- Submission Type: {submission_type}
- Regulatory Pathway: {regulatory_pathway}
- Known Predicates: {known_predicates}
- Clinical Data Available: {clinical_data_status}
- Special Circumstances: {special_circumstances}

**REQUESTED ANALYSIS DEPTH:**
- {depth_quick}
- {depth_standard}
- {depth_comprehensive}

Based on this information, please provide:

1. **Device Classification Analysis** - Determine device type, CFR part, risk class
2. **Comprehensive Agent Recommendation** - All applicable agents organized by phase
3. **Execution Sequence** - Optimal order and parallelization opportunities
4. **Timeline Estimate** - Realistic review timeline
5. **Critical Focus Areas** - What reviewers must pay special attention to
6. **Anticipated Challenges** - Potential regulatory hurdles
7. **Execution Commands** - Ready-to-use commands for running agents

If any device information is unclear or incomplete, please note what additional 
details would help refine your recommendations.

Use only agents that exist in the provided agents.yaml catalog. 
Return the orchestration plan in well-structured markdown with all required tables.
"""

# =========================
# Dynamic Agent Generator Prompt
# =========================

DYNAMIC_AGENT_SYSTEM_PROMPT = """
You are a regulatory agent designer. Your job is to look at FDA guidance content
(and optionally an existing review checklist) and generate NEW specialized review
agents in YAML format that can be added to agents.yaml.

Requirements:

1. Output valid YAML defining between 3 and 8 agents under a top-level key `agents`.
2. Each agent must include at least:
   - agent_id: string (e.g., "AGENT-200")
   - name: descriptive name
   - version: "1.0"
   - category: a high-level category such as "Core Review", "Specialized Analysis",
     "Device-Specific Expert", or "Workflow Utility"
   - description: 1–2 sentence description of what the agent does
   - model: pick a realistic default (e.g., "gpt-4o-mini" or "gemini-2.5-flash")
   - temperature: float (0.1–0.4 for deterministic regulatory work)
   - max_tokens: integer (e.g., 16000–22000)
   - system_prompt: detailed instructions for the agent, grounded in the guidance
   - user_prompt_template: with placeholders for user inputs
   - output_requirements: structured object describing min tables/sections/etc.
   - validation_rules: simple checks (e.g., require_sections, check_table_count)

3. Agents must be clearly tied to sections of the FDA guidance. For example:
   - A "Performance Testing Matrix Builder" agent that converts guidance testing
     requirements into a matrix.
   - A "Benefit-Risk Focused Reviewer" agent for guidance sections on benefit-risk.
   - A "Human Factors & Usability Reviewer" agent for HF-focused guidance.

4. Do NOT repeat agents that already exist in the provided agents.yaml catalog.
   Create complementary, non-duplicative capabilities.

5. Use descriptions and prompt templates similar in richness and style to the
   existing default agents (e.g., intelligence_analyst, guidance_to_checklist_converter).

Return ONLY YAML. Do not wrap it in markdown backticks.
"""

# =========================
# Agents YAML standardization prompt
# =========================

STANDARDIZE_AGENTS_SYSTEM_PROMPT = """
You are an expert YAML normalizer for an agents.yaml catalog.

Goal:
- Take possibly messy or partial YAML / text that describes agents.
- Optionally, the user may also provide an example of the desired standardized format.
- Produce a CLEAN, VALID YAML document with a top-level key `agents`,
  where each child key is an agent_id and each agent has, at minimum:
  - agent_id
  - name
  - version
  - category
  - description
  - model
  - temperature
  - max_tokens
  - system_prompt
  - user_prompt_template
  - output_requirements
  - validation_rules

Rules:
- Output MUST be valid YAML. DO NOT wrap in markdown code fences.
- If the input already contains YAML, normalize and complete missing fields.
- If the input is partially structured text, infer reasonable fields based on content.
- Use the same key names as above. You may add extra fields, but never remove these.
- Prefer using models from: gemini-2.5-flash, gemini-3-pro-preview, gpt-4o-mini, gpt-5-mini.
- Keep textual content (name, description, prompts) in Traditional Chinese where appropriate,
  but YAML keys stay in English as specified.
"""


# =========================
# Helper functions
# =========================

def t(key: str) -> str:
    """Translate label key based on current language."""
    lang = st.session_state.settings.get("language", "English")
    return LABELS.get(key, {}).get(lang, key)


def apply_style(theme: str, painter_style: str):
    """Apply painter-based WOW CSS and theme adjustments."""
    css = STYLE_CSS.get(painter_style, "")
    if theme == "Dark":
        css += """
          body { color: #e0e0e0; }
          .stButton>button { background-color: #1f2933; color: white; border-radius: 999px; }
          .stTextInput>div>div>input, .stTextArea textarea {
            background-color: #111827; color: #e5e7eb; border-radius: 0.5rem;
          }
        """
    else:
        css += """
          body { color: #111827; }
          .stButton>button { background-color: #2563eb; color: white; border-radius: 999px; }
          .stTextInput>div>div>input, .stTextArea textarea {
            background-color: #ffffff; color: #111827; border-radius: 0.5rem;
          }
        """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def get_provider(model: str) -> str:
    if model in OPENAI_MODELS:
        return "openai"
    if model in GEMINI_MODELS:
        return "gemini"
    if model in ANTHROPIC_MODELS:
        return "anthropic"
    if model in GROK_MODELS:
        return "grok"
    raise ValueError(f"Unknown model: {model}")


def call_llm(model: str, system_prompt: str, user_prompt: str,
             max_tokens: int = 12000, temperature: float = 0.2,
             api_keys: dict | None = None) -> str:
    """Synchronous LLM call with routing across OpenAI, Gemini, Anthropic, Grok."""
    provider = get_provider(model)
    api_keys = api_keys or {}

    def get_key(name: str, env_var: str) -> str:
        return api_keys.get(name) or os.getenv(env_var) or ""

    if provider == "openai":
        key = get_key("openai", "OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Missing OpenAI API key.")
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    if provider == "gemini":
        key = get_key("gemini", "GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Missing Gemini API key.")
        genai.configure(api_key=key)
        llm = genai.GenerativeModel(model)
        resp = llm.generate_content(
            system_prompt + "\n\n" + user_prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
        )
        return resp.text

    if provider == "anthropic":
        key = get_key("anthropic", "ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Missing Anthropic API key.")
        client = Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text

    if provider == "grok":
        key = get_key("grok", "GROK_API_KEY")
        if not key:
            raise RuntimeError("Missing Grok (xAI) API key.")
        with httpx.Client(base_url="https://api.x.ai/v1", timeout=60) as client:
            resp = client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]


def show_status(step_name: str, status: str):
    """Small colored indicator."""
    color = {
        "pending": "gray",
        "running": "#f59e0b",
        "done": "#10b981",
        "error": "#ef4444",
    }.get(status, "gray")
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;margin-bottom:0.25rem;">
          <div style="width:10px;height:10px;border-radius:50%;background:{color};margin-right:6px;"></div>
          <span style="font-size:0.9rem;">{step_name} – <b>{status}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def log_event(tab: str, agent: str, model: str, tokens_est: int):
    st.session_state["history"].append({
        "tab": tab,
        "agent": agent,
        "model": model,
        "tokens_est": tokens_est,
        "ts": datetime.utcnow().isoformat()
    })


def extract_pdf_pages_to_text(file, start_page: int, end_page: int) -> str:
    """Extract text from a PDF between start_page and end_page (1-based, inclusive)."""
    reader = PdfReader(file)
    n = len(reader.pages)
    start = max(0, start_page - 1)
    end = min(n, end_page)
    texts = []
    for i in range(start, end):
        try:
            texts.append(reader.pages[i].extract_text() or "")
        except Exception:
            texts.append("")
    return "\n\n".join(texts)


def extract_docx_to_text(file) -> str:
    """Extract text from a DOCX file (if python-docx is installed)."""
    if Document is None:
        return ""
    try:
        bio = BytesIO(file.read())
        doc = Document(bio)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""


def create_pdf_from_text(text: str) -> bytes:
    """Create a simple multi-page PDF from plain text using reportlab."""
    if canvas is None or letter is None:
        raise RuntimeError(
            "PDF generation library 'reportlab' is not installed. "
            "Please add 'reportlab' to your Space requirements."
        )

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = 72  # 1 inch
    line_height = 14
    y = height - margin

    for line in text.splitlines():
        if y < margin:
            c.showPage()
            y = height - margin
        c.drawString(margin, y, line[:2000])
        y -= line_height

    c.save()
    buf.seek(0)
    return buf.getvalue()


def convert_office_to_pdf(uploaded_file) -> tuple[bytes, int]:
    """
    Convert DOCX/DOC to PDF bytes.
    - DOCX: via python-docx
    - DOC: best-effort; may require extra libs (textract).
    Returns (pdf_bytes, page_count).
    """
    filename = uploaded_file.name or "document"
    suffix = filename.lower().rsplit(".", 1)[-1]
    file_bytes = uploaded_file.read()

    text = ""

    if suffix == "docx":
        if Document is None:
            raise RuntimeError(
                "DOCX conversion requires 'python-docx', which is not installed."
            )
        text = extract_docx_to_text(BytesIO(file_bytes))

    elif suffix == "doc":
        try:
            import textract  # optional
            text = textract.process(BytesIO(file_bytes)).decode("utf-8", errors="ignore")
        except Exception:
            raise RuntimeError(
                "DOC (legacy Word) conversion requires extra tooling not available here. "
                "請先將 .doc 轉為 .docx 或 PDF 後再上傳。"
            )
    else:
        raise RuntimeError("Only DOCX and DOC files are supported in this converter.")

    if not text.strip():
        raise RuntimeError("No extractable text found in the uploaded document.")

    pdf_bytes = create_pdf_from_text(text)
    reader = PdfReader(BytesIO(pdf_bytes))
    page_count = len(reader.pages)
    return pdf_bytes, page_count


def show_pdf(pdf_bytes: bytes, height: int = 600):
    """Inline PDF preview using an iframe."""
    if not pdf_bytes:
        return
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_html = f"""
        <iframe
            src="data:application/pdf;base64,{b64}"
            width="100%"
            height="{height}"
            type="application/pdf">
        </iframe>
    """
    st.markdown(pdf_html, unsafe_allow_html=True)


def agent_run_ui(
    agent_id: str,
    tab_key: str,
    default_prompt: str,
    default_input_text: str = "",
    allow_model_override: bool = True,
    tab_label_for_history: str | None = None,
):
    """Reusable UI for running any agent defined in agents.yaml."""
    agents_cfg = st.session_state["agents_cfg"]
    agent_cfg = agents_cfg["agents"][agent_id]
    status_key = f"{tab_key}_status"
    if status_key not in st.session_state:
        st.session_state[status_key] = "pending"

    show_status(agent_cfg.get("name", agent_id), st.session_state[status_key])

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        user_prompt = st.text_area(
            "Prompt",
            value=st.session_state.get(f"{tab_key}_prompt", default_prompt),
            height=160,
            key=f"{tab_key}_prompt",
        )
    with col2:
        base_model = agent_cfg.get("model", st.session_state.settings["model"])
        model_index = ALL_MODELS.index(base_model) if base_model in ALL_MODELS else 0
        model = st.selectbox(
            "Model",
            ALL_MODELS,
            index=model_index,
            disabled=not allow_model_override,
            key=f"{tab_key}_model",
        )
    with col3:
        max_tokens = st.number_input(
            "max_tokens",
            min_value=1000,
            max_value=120000,
            value=st.session_state.settings["max_tokens"],
            step=1000,
            key=f"{tab_key}_max_tokens",
        )

    input_text = st.text_area(
        "Input Text / Markdown",
        value=st.session_state.get(f"{tab_key}_input", default_input_text),
        height=260,
        key=f"{tab_key}_input",
    )

    run = st.button("Run Agent", key=f"{tab_key}_run")

    if run:
        st.session_state[status_key] = "running"
        show_status(agent_cfg.get("name", agent_id), "running")
        api_keys = st.session_state.get("api_keys", {})
        system_prompt = agent_cfg.get("system_prompt", "")
        user_full = f"{user_prompt}\n\n---\n\n{input_text}"

        with st.spinner("Running agent..."):
            try:
                out = call_llm(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_full,
                    max_tokens=max_tokens,
                    temperature=st.session_state.settings["temperature"],
                    api_keys=api_keys,
                )
                st.session_state[f"{tab_key}_output"] = out
                st.session_state[status_key] = "done"
                token_est = int(len(user_full + out) / 4)
                log_event(
                    tab_label_for_history or tab_key,
                    agent_cfg.get("name", agent_id),
                    model,
                    token_est,
                )
            except Exception as e:
                st.session_state[status_key] = "error"
                st.error(f"Agent error: {e}")

    # Editable output
    output = st.session_state.get(f"{tab_key}_output", "")
    view_mode = st.radio(
        "View mode", ["Markdown", "Plain text"],
        horizontal=True, key=f"{tab_key}_viewmode"
    )
    if view_mode == "Markdown":
        edited = st.text_area(
            "Output (Markdown, editable)",
            value=output,
            height=320,
            key=f"{tab_key}_output_md",
        )
    else:
        edited = st.text_area(
            "Output (Plain text, editable)",
            value=output,
            height=320,
            key=f"{tab_key}_output_txt",
        )

    st.session_state[f"{tab_key}_output_edited"] = edited


# =========================
# 510(k) Summary JSON schema (prompt-driven)
# =========================

FDA_510K_JSON_SCHEMA_DESCRIPTION = """
You must return a SINGLE JSON object with the following top-level keys:

- "metadata": {
    "k_number": string,
    "submission_type": string,
    "decision_date": string,
    "device_name": string,
    "regulation_number": string,
    "product_code": string,
    "review_panel": string
  }

- "submitter": {
    "name": string,
    "address": string,
    "contact_person": string,
    "contact_details": string
  }

- "device_description": {
    "trade_name": string,
    "common_name": string,
    "classification_name": string,
    "device_class": string,
    "regulation_number": string,
    "panel": string,
    "technology_overview": string,
    "design_features": [string],
    "materials": [string]
  }

- "indications_for_use": {
    "indications_statement": string,
    "intended_use": string,
    "patient_population": string,
    "anatomical_site": string,
    "environment_of_use": string,
    "contraindications": [string],
    "warnings": [string],
    "precautions": [string]
  }

- "predicate_devices": [
    {
      "k_number": string,
      "device_name": string,
      "manufacturer": string,
      "regulation_number": string,
      "product_code": string,
      "similarities": [string],
      "differences": [string],
      "impact_of_differences": string
    }
  ]

- "performance_testing": {
    "bench": [
      {
        "name": string,
        "purpose": string,
        "standard_or_method": string,
        "acceptance_criteria": string,
        "result": string,
        "summary": string
      }
    ],
    "biocompatibility": [
      {
        "name": string,
        "standard": string,
        "endpoints": [string],
        "result": string
      }
    ],
    "software_and_cybersecurity": [
      {
        "name": string,
        "software_level": string,
        "standard_or_method": string,
        "result": string
      }
    ],
    "animal": [
      {
        "name": string,
        "model": string,
        "endpoints": [string],
        "duration": string,
        "result": string
      }
    ],
    "clinical": [
      {
        "name": string,
        "design": string,
        "population": string,
        "primary_endpoints": [string],
        "key_results": string
      }
    ],
    "standards": [
      {
        "identifier": string,
        "title": string,
        "type": string,
        "status": string
      }
    ]
  }

- "risk_management": {
    "primary_risks": [
      {
        "risk": string,
        "hazard": string,
        "clinical_effect": string,
        "mitigations": [string]
      }
    ],
    "benefit_risk_summary": string
  }

- "substantial_equivalence": {
    "conclusion": string,
    "key_arguments": [string]
  }

- "document_quality": {
    "model_uncertainty": string,
    "missing_information": [string],
    "assumptions": [string]
  }

Every string field may be empty if not found. Arrays may be empty, but all keys must exist.
Do not add extra top-level keys.
"""


def parse_510k_summary_with_gemini(
    text: str,
    model: str,
    api_keys: dict,
    max_tokens: int = 6000,
    temperature: float = 0.1,
) -> dict:
    """
    Use Gemini (gemini-2.5-flash or gemini-3-pro-preview) to parse an FDA 510(k)
    summary (text or markdown) into a structured JSON object following
    FDA_510K_JSON_SCHEMA_DESCRIPTION.
    """

    if model not in {"gemini-2.5-flash", "gemini-3-pro-preview"}:
        raise ValueError("510(k) Summary parsing requires a Gemini model (gemini-2.5-flash or gemini-3-pro-preview).")

    key = api_keys.get("gemini") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Missing Gemini API key for 510(k) Summary parsing.")

    genai.configure(api_key=key)

    system_prompt = f"""
You are an FDA 510(k) regulatory analyst.

Your job:
1. Read the provided 510(k) summary (unstructured text or markdown).
2. Extract key regulatory and technical information.
3. Return a SINGLE JSON object that STRICTLY conforms to this schema:

{FDA_510K_JSON_SCHEMA_DESCRIPTION}

Rules:
- Do NOT invent details that clearly conflict with the text.
- If information is not present, leave the corresponding field as an empty string or
  an empty array, but keep the key.
- Output MUST be pure JSON (no markdown, no commentary).
"""

    llm = genai.GenerativeModel(model)
    resp = llm.generate_content(
        [system_prompt, text],
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "response_mime_type": "application/json",
        },
    )

    raw = resp.text or ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw[start:end + 1])
            except Exception:
                raise RuntimeError("Gemini did not return valid JSON; please try again with a smaller input.")
        else:
            raise RuntimeError("Gemini did not return recognizable JSON; please try again with a smaller input.")

    if not isinstance(data, dict):
        raise RuntimeError("Parsed 510(k) data is not a JSON object as expected.")

    return data


def render_510k_summary_dashboard(structured: dict):
    """
    Visual 510(k) summary dashboard:
    - KPI cards for metadata
    - Collapsible sections
    - Tables and simple infographics
    """
    # Local CSS: teal / slate / white "medical grade" styling
    st.markdown(
        """
        <style>
        .k510-hero {
            background: linear-gradient(135deg, #0f766e, #0f172a);
            border-radius: 18px;
            padding: 18px 22px;
            color: #ecfeff;
            margin-bottom: 1rem;
        }
        .k510-hero-title {
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            color: #a5f3fc;
        }
        .k510-hero-main {
            font-size: 1.5rem;
            font-weight: 700;
        }
        .k510-kpi-card {
            background: #0b1120;
            border-radius: 14px;
            padding: 12px 14px;
            color: #e5e7eb;
            border: 1px solid #1f2937;
            box-shadow: 0 6px 16px rgba(15,23,42,0.35);
        }
        .k510-kpi-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #9ca3af;
        }
        .k510-kpi-value {
            font-size: 1.1rem;
            font-weight: 600;
            color: #ecfeff;
        }
        .k510-section-title {
            font-weight: 700;
            font-size: 1.0rem;
            color: #0f172a;
        }
        .k510-section-subtitle {
            font-size: 0.9rem;
            color: #4b5563;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    metadata = structured.get("metadata", {}) or {}
    submitter = structured.get("submitter", {}) or {}
    device_desc = structured.get("device_description", {}) or {}
    indications = structured.get("indications_for_use", {}) or {}
    predicates = structured.get("predicate_devices", []) or []
    perf = structured.get("performance_testing", {}) or {}
    risk = structured.get("risk_management", {}) or {}
    se = structured.get("substantial_equivalence", {}) or {}
    doc_quality = structured.get("document_quality", {}) or {}

    # Hero
    device_name = metadata.get("device_name") or device_desc.get("trade_name") or "Unnamed Device"
    k_number = metadata.get("k_number") or "Unknown 510(k)"
    submission_type = metadata.get("submission_type") or "N/A"
    decision_date = metadata.get("decision_date") or "N/A"

    st.markdown(
        f"""
        <div class="k510-hero">
          <div class="k510-hero-title">FDA 510(k) Summary Intelligence</div>
          <div class="k510-hero-main">{device_name}</div>
          <div style="margin-top:0.25rem;font-size:0.9rem;color:#e5e7eb;">
            510(k) Number: <b>{k_number}</b> &nbsp;&nbsp; | &nbsp;&nbsp;
            Submission type: <b>{submission_type}</b> &nbsp;&nbsp; | &nbsp;&nbsp;
            Decision date: <b>{decision_date}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="k510-kpi-card">
              <div class="k510-kpi-label">Product Code</div>
              <div class="k510-kpi-value">{metadata.get("product_code") or "—"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="k510-kpi-card">
              <div class="k510-kpi-label">Regulation</div>
              <div class="k510-kpi-value">{metadata.get("regulation_number") or device_desc.get("regulation_number") or "—"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="k510-kpi-card">
              <div class="k510-kpi-label">Classification</div>
              <div class="k510-kpi-value">{device_desc.get("device_class") or "—"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="k510-kpi-card">
              <div class="k510-kpi-label">Review Panel</div>
              <div class="k510-kpi-value">{metadata.get("review_panel") or device_desc.get("panel") or "—"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Collapsible sections
    with st.expander("Submitter & Device Overview", expanded=True):
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Submitter Information**")
            df_submitter = pd.DataFrame(
                [
                    ["Name", submitter.get("name") or ""],
                    ["Address", submitter.get("address") or ""],
                    ["Contact person", submitter.get("contact_person") or ""],
                    ["Contact details", submitter.get("contact_details") or ""],
                ],
                columns=["Field", "Value"],
            )
            st.table(df_submitter)

        with colB:
            st.markdown("**Device Description**")
            df_device = pd.DataFrame(
                [
                    ["Trade name", device_desc.get("trade_name") or ""],
                    ["Common name", device_desc.get("common_name") or ""],
                    ["Classification name", device_desc.get("classification_name") or ""],
                    ["Device class", device_desc.get("device_class") or ""],
                    ["Regulation number", device_desc.get("regulation_number") or ""],
                    ["Panel", device_desc.get("panel") or ""],
                ],
                columns=["Field", "Value"],
            )
            st.table(df_device)

            if device_desc.get("technology_overview"):
                st.markdown("**Technology overview**")
                st.markdown(device_desc["technology_overview"])

            if device_desc.get("design_features"):
                st.markdown("**Key design features**")
                st.markdown("- " + "\n- ".join(device_desc["design_features"]))

    with st.expander("Indications for Use & Clinical Context", expanded=True):
        st.markdown("**Indications for Use Statement**")
        st.markdown(indications.get("indications_statement") or "_Not explicitly stated in summary._")

        colI1, colI2 = st.columns(2)
        with colI1:
            st.markdown("**Intended Use & Patient Population**")
            df_ind = pd.DataFrame(
                [
                    ["Intended use", indications.get("intended_use") or ""],
                    ["Patient population", indications.get("patient_population") or ""],
                    ["Anatomical site", indications.get("anatomical_site") or ""],
                    ["Environment of use", indications.get("environment_of_use") or ""],
                ],
                columns=["Aspect", "Details"],
            )
            st.table(df_ind)
        with colI2:
            st.markdown("**Contraindications / Warnings / Precautions**")
            st.markdown("**Contraindications**")
            st.markdown(
                "- " + "\n- ".join(indications.get("contraindications") or ["Not specified."])
            )
            st.markdown("**Warnings**")
            st.markdown(
                "- " + "\n- ".join(indications.get("warnings") or ["Not specified."])
            )
            st.markdown("**Precautions**")
            st.markdown(
                "- " + "\n- ".join(indications.get("precautions") or ["Not specified."])
            )

    with st.expander("Predicate Devices & Substantial Equivalence", expanded=True):
        if predicates:
            rows = []
            for p in predicates:
                rows.append(
                    {
                        "510(k)": p.get("k_number", ""),
                        "Device name": p.get("device_name", ""),
                        "Manufacturer": p.get("manufacturer", ""),
                        "Regulation": p.get("regulation_number", ""),
                        "Product code": p.get("product_code", ""),
                    }
                )
            df_pred = pd.DataFrame(rows)
            st.markdown("**Predicate Device Table**")
            st.table(df_pred)

            # Simple bar chart: how many predicates by manufacturer
            try:
                if "Manufacturer" in df_pred.columns and not df_pred["Manufacturer"].isna().all():
                    chart = (
                        alt.Chart(df_pred)
                        .mark_bar()
                        .encode(
                            x=alt.X("Manufacturer:N", sort="-y"),
                            y="count():Q",
                            color="Manufacturer:N",
                            tooltip=["Manufacturer", "count()"],
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(chart, use_container_width=True)
            except Exception:
                pass

            # Similarities/differences narrative
            for idx, p in enumerate(predicates, start=1):
                st.markdown(f"**Predicate #{idx} – {p.get('k_number','')}**")
                if p.get("similarities"):
                    st.markdown("_Key similarities:_")
                    st.markdown("- " + "\n- ".join(p["similarities"]))
                if p.get("differences"):
                    st.markdown("_Key differences:_")
                    st.markdown("- " + "\n- ".join(p["differences"]))
                if p.get("impact_of_differences"):
                    st.markdown("**Impact of differences**")
                    st.markdown(p["impact_of_differences"])
                st.markdown("---")
        else:
            st.info("No predicate device details extracted from the summary.")

        st.markdown("**Substantial Equivalence Conclusion**")
        st.markdown(se.get("conclusion") or "_No explicit SE conclusion captured._")
        if se.get("key_arguments"):
            st.markdown("**Key SE arguments**")
            st.markdown("- " + "\n- ".join(se["key_arguments"]))

    with st.expander("Performance Testing Landscape", expanded=True):
        perf_rows = []
        for cat_name, cat_label in [
            ("bench", "Bench"),
            ("biocompatibility", "Biocompatibility"),
            ("software_and_cybersecurity", "Software / Cybersecurity"),
            ("animal", "Animal"),
            ("clinical", "Clinical"),
        ]:
            tests = perf.get(cat_name, []) or []
            for t in tests:
                if isinstance(t, dict):
                    perf_rows.append(
                        {
                            "Category": cat_label,
                            "Name": str(t.get("name", "")),
                            "Result": str(t.get("result", t.get("key_results", ""))),
                        }
                    )

        if perf_rows:
            df_perf = pd.DataFrame(perf_rows)
            st.markdown("**Performance testing summary table**")
            st.table(df_perf)

            try:
                chart_counts = (
                    alt.Chart(df_perf)
                    .mark_bar()
                    .encode(
                        x=alt.X("Category:N", sort="-y"),
                        y="count():Q",
                        color="Category:N",
                        tooltip=["Category", "count()"],
                    )
                    .properties(height=220, title="Number of tests by category")
                )
                st.altair_chart(chart_counts, use_container_width=True)
            except Exception:
                pass
        else:
            st.info("No detailed performance testing structure extracted.")

        standards = perf.get("standards", []) or []
        if standards:
            df_std = pd.DataFrame(standards)
            st.markdown("**Recognized standards used**")
            st.table(df_std)

    with st.expander("Risk Management & Benefit–Risk Profile", expanded=False):
        primary_risks = risk.get("primary_risks", []) or []
        if primary_risks:
            rows = []
            for r in primary_risks:
                rows.append(
                    {
                        "Risk": r.get("risk", ""),
                        "Hazard": r.get("hazard", ""),
                        "Clinical effect": r.get("clinical_effect", ""),
                        "Mitigations": "; ".join(r.get("mitigations", []) or []),
                    }
                )
            df_risk = pd.DataFrame(rows)
            st.markdown("**Risk–mitigation matrix**")
            st.table(df_risk)
        else:
            st.info("No structured risk–mitigation matrix extracted.")

        st.markdown("**Benefit–risk summary**")
        st.markdown(risk.get("benefit_risk_summary") or "_No explicit benefit–risk narrative captured._")

    with st.expander("Document Quality & Model Caveats", expanded=False):
        st.markdown("**Model Uncertainty**")
        st.markdown(doc_quality.get("model_uncertainty") or "_No explicit model uncertainty text._")

        if doc_quality.get("missing_information"):
            st.markdown("**Potential missing information from the 510(k) summary**")
            st.markdown("- " + "\n- ".join(doc_quality["missing_information"]))

        if doc_quality.get("assumptions"):
            st.markdown("**Model assumptions while structuring this summary**")
            st.markdown("- " + "\n- ".join(doc_quality["assumptions"]))


def render_510k_contextual_chat(structured: dict, original_text: str):
    """
    Right-hand side contextual chat, grounded in:
    - structured JSON
    - original 510(k) summary text

    Model choices: gemini-2.5-flash (faster) or gemini-3-pro-preview (deeper).
    """
    if "k510_chat_history" not in st.session_state:
        st.session_state["k510_chat_history"] = []

    st.markdown("#### Contextual Q&A on This 510(k) Summary")

    chat_model = st.selectbox(
        "Chat model",
        ["gemini-2.5-flash", "gemini-3-pro-preview"],
        index=0,
        key="k510_chat_model",
        help="gemini-2.5-flash = faster; gemini-3-pro-preview = deeper reasoning",
    )

    # Display chat history
    for idx, msg in enumerate(st.session_state["k510_chat_history"]):
        role = msg["role"]
        content = msg["content"]
        align = "flex-start" if role == "user" else "flex-end"
        bg = "#e0f2fe" if role == "user" else "#ecfdf5"
        border = "#38bdf8" if role == "user" else "#22c55e"
        st.markdown(
            f"""
            <div style="display:flex;justify-content:{align};margin-bottom:0.3rem;">
              <div style="max-width:100%;background:{bg};border:1px solid {border};padding:8px 10px;border-radius:10px;font-size:0.9rem;">
                <b>{'You' if role=='user' else 'Assistant'}</b><br>{content}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    user_q = st.text_area(
        "Ask a question about this specific 510(k) summary",
        height=100,
        key="k510_chat_input",
    )

    if st.button("Send question", key="k510_chat_send"):
        if not user_q.strip():
            st.warning("Please type a question before sending.")
            return

        st.session_state["k510_chat_history"].append({"role": "user", "content": user_q})

        api_keys = st.session_state.get("api_keys", {})
        structured_json_str = json.dumps(structured, ensure_ascii=False, indent=2)

        system_prompt = f"""
You are a specialized FDA 510(k) review assistant.

You are given:
1. A structured JSON representation of a 510(k) summary:
{structured_json_str}

2. The original unstructured 510(k) summary text (may contain additional nuance):
\"\"\"{original_text[:12000]}\"\"\"  # only first ~12k chars to avoid overlong context

Your tasks when answering questions:
- Use the JSON as the primary reference for facts.
- Use the raw text to add nuance or clarify ambiguous areas.
- Always answer as if you are explaining to a regulatory affairs professional.
- If the user asks for something not supported by the document, say so explicitly.
- Do not hallucinate predicates, performance tests, or claims not found in the JSON or text.
"""

        last_turns = st.session_state["k510_chat_history"][-6:]  # last 6 messages max
        convo_text = []
        for m in last_turns:
            convo_text.append(f"{m['role'].upper()}: {m['content']}")
        convo_joined = "\n\n".join(convo_text)

        with st.spinner("Generating answer from the 510(k) summary..."):
            try:
                answer = call_llm(
                    model=chat_model,
                    system_prompt=system_prompt,
                    user_prompt=convo_joined,
                    max_tokens=4000,
                    temperature=0.2,
                    api_keys=api_keys,
                )
            except Exception as e:
                st.error(f"Chat error: {e}")
                return

        st.session_state["k510_chat_history"].append(
            {"role": "assistant", "content": answer}
        )
        token_est = int(len(convo_joined + answer) / 4)
        log_event(
            "510(k) Summary Chat",
            "510(k) Contextual QA",
            chat_model,
            token_est,
        )
        st.experimental_rerun()


# =========================
# Sidebar (WOW UI + API)
# =========================

def render_sidebar():
    with st.sidebar:
        st.markdown("### Global Settings")

        # Theme
        st.session_state.settings["theme"] = st.radio(
            "Theme", ["Light", "Dark"],
            index=0 if st.session_state.settings["theme"] == "Light" else 1,
        )

        # Language
        st.session_state.settings["language"] = st.radio(
            "Language", ["English", "繁體中文"],
            index=0 if st.session_state.settings["language"] == "English" else 1,
        )

        # Painter style + Jackpot
        col1, col2 = st.columns([3, 1])
        with col1:
            style = st.selectbox(
                "Painter Style",
                PAINTER_STYLES,
                index=PAINTER_STYLES.index(st.session_state.settings["painter_style"]),
            )
        with col2:
            if st.button("Jackpot!"):
                import random
                style = random.choice(PAINTER_STYLES)
        st.session_state.settings["painter_style"] = style

        # Default model, tokens, temperature
        st.session_state.settings["model"] = st.selectbox(
            "Default Model",
            ALL_MODELS,
            index=ALL_MODELS.index(st.session_state.settings["model"]),
        )
        st.session_state.settings["max_tokens"] = st.number_input(
            "Default max_tokens",
            min_value=1000,
            max_value=120000,
            value=st.session_state.settings["max_tokens"],
            step=1000,
        )
        st.session_state.settings["temperature"] = st.slider(
            "Temperature",
            0.0,
            1.0,
            st.session_state.settings["temperature"],
            0.05,
        )

        # API Keys
        st.markdown("---")
        st.markdown("### API Keys")

        keys = {}

        if os.getenv("OPENAI_API_KEY"):
            keys["openai"] = os.getenv("OPENAI_API_KEY")
            st.caption("OpenAI key from environment.")
        else:
            keys["openai"] = st.text_input("OpenAI API Key", type="password")

        if os.getenv("GEMINI_API_KEY"):
            keys["gemini"] = os.getenv("GEMINI_API_KEY")
            st.caption("Gemini key from environment.")
        else:
            keys["gemini"] = st.text_input("Gemini API Key", type="password")

        if os.getenv("ANTHROPIC_API_KEY"):
            keys["anthropic"] = os.getenv("ANTHROPIC_API_KEY")
            st.caption("Anthropic key from environment.")
        else:
            keys["anthropic"] = st.text_input("Anthropic API Key", type="password")

        if os.getenv("GROK_API_KEY"):
            keys["grok"] = os.getenv("GROK_API_KEY")
            st.caption("Grok key from environment.")
        else:
            keys["grok"] = st.text_input("Grok API Key", type="password")

        st.session_state["api_keys"] = keys

        # Optional override of agents.yaml (still supported here)
        st.markdown("---")
        st.markdown("### Agents Catalog (agents.yaml)")
        uploaded_agents = st.file_uploader(
            "Upload custom agents.yaml",
            type=["yaml", "yml"],
            key="sidebar_agents_yaml",
        )
        if uploaded_agents is not None:
            try:
                cfg = yaml.safe_load(uploaded_agents.read())
                if "agents" in cfg:
                    st.session_state["agents_cfg"] = cfg
                    st.success("Custom agents.yaml loaded for this session.")
                else:
                    st.warning("Uploaded YAML has no top-level 'agents' key. Using previous config.")
            except Exception as e:
                st.error(f"Failed to parse uploaded YAML: {e}")


# =========================
# Tab renderers
# =========================

def render_dashboard():
    st.title(t("Dashboard"))
    hist = st.session_state["history"]
    if not hist:
        st.info("No runs yet.")
        return

    df = pd.DataFrame(hist)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Runs", len(df))
    with col2:
        st.metric("Unique 510(k) Sessions", df[df["tab"].str.contains("510", na=False)].shape[0])
    with col3:
        st.metric("Approx Tokens Processed", int(df["tokens_est"].sum()))

    st.subheader("Runs by Tab")
    chart_tab = alt.Chart(df).mark_bar().encode(
        x="tab:N",
        y="count():Q",
        color="tab:N",
        tooltip=["tab", "count()"],
    )
    st.altair_chart(chart_tab, use_container_width=True)

    st.subheader("Runs by Model")
    chart_model = alt.Chart(df).mark_bar().encode(
        x="model:N",
        y="count():Q",
        color="model:N",
        tooltip=["model", "count()"],
    )
    st.altair_chart(chart_model, use_container_width=True)

    st.subheader("Token Usage Over Time")
    df_time = df.copy()
    df_time["ts"] = pd.to_datetime(df_time["ts"])
    chart_time = alt.Chart(df_time).mark_line(point=True).encode(
        x="ts:T",
        y="tokens_est:Q",
        color="tab:N",
        tooltip=["ts", "tab", "agent", "model", "tokens_est"],
    )
    st.altair_chart(chart_time, use_container_width=True)

    st.subheader("Recent Activity")
    st.dataframe(df.sort_values("ts", ascending=False).head(25), use_container_width=True)


def render_510k_tab():
    st.title(t("510k_tab"))

    col1, col2 = st.columns(2)
    with col1:
        device_name = st.text_input("Device Name")
        k_number = st.text_input("510(k) Number (e.g., K123456)")
    with col2:
        sponsor = st.text_input("Sponsor / Manufacturer (optional)")
        product_code = st.text_input("Product Code (optional)")

    extra_info = st.text_area("Additional context (indications, technology, etc.)")

    default_prompt = f"""
You are assisting an FDA 510(k) reviewer.

Task:
1. Search FDA resources (or emulate such search) for:
   - Device: {device_name}
   - 510(k) number: {k_number}
   - Sponsor: {sponsor}
   - Product code: {product_code}
2. Synthesize a 3000–4000 word detailed, review-oriented summary.
3. Provide AT LEAST 5 well-structured markdown tables covering at minimum:
   - Device overview (trade name, sponsor, 510(k) number, product code, regulation number)
   - Indications for use and intended population
   - Technological characteristics and comparison with predicate(s)
   - Performance testing (bench, animal, clinical) and acceptance criteria
   - Risks and corresponding risk controls/mitigations

Language: {st.session_state.settings["language"]}.

Use headings that match FDA 510(k) review style.
"""
    combined_input = f"""
=== Device Inputs ===
Device name: {device_name}
510(k) number: {k_number}
Sponsor: {sponsor}
Product code: {product_code}

Additional context:
{extra_info}
"""

    agent_run_ui(
        agent_id="fda_search_agent",
        tab_key="510k",
        default_prompt=default_prompt,
        default_input_text=combined_input,
        tab_label_for_history="510(k) Intelligence",
    )


def render_510k_summary_studio_tab():
    """
    New WOW tab:
    - Users can paste text/markdown or upload a file (pdf, txt, md, json) of an FDA 510(k) summary.
    - System parses it with Gemini into structured JSON.
    - Left: interactive dashboard with infographics & tables.
    - Right: contextual chat grounded in that specific document.
    """
    st.title(t("510k_summary_studio"))

    if "k510_status" not in st.session_state:
        st.session_state["k510_status"] = "pending"

    show_status("510(k) Summary Analysis", st.session_state["k510_status"])

    st.markdown(
        """
        <div style="background:#f0fdf4;border-radius:12px;padding:12px 14px;border:1px solid #bbf7d0;margin-bottom:0.75rem;">
          <b>Step 1.</b> Paste or upload an FDA 510(k) summary (text, markdown, or file).<br>
          <b>Step 2.</b> Click <i>Analyze 510(k) Summary</i> to create a structured regulatory dashboard.<br>
          <b>Step 3.</b> Use the right-side chat to ask regulatory questions about this specific summary.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Input mode: paste or upload ---
    col_input_left, col_input_right = st.columns([3, 2])

    with col_input_left:
        input_mode = st.radio(
            "Input mode",
            ["Paste text / markdown", "Upload file (PDF/TXT/MD/JSON)"],
            horizontal=True,
            key="k510_input_mode",
        )

        raw_text = ""
        uploaded_structured = None

        if input_mode == "Paste text / markdown":
            raw_text = st.text_area(
                "Paste 510(k) summary (text or markdown)",
                height=260,
                key="k510_paste_text",
            )
        else:
            up = st.file_uploader(
                "Upload 510(k) summary file",
                type=["pdf", "txt", "md", "json"],
                key="k510_file_uploader",
            )
            if up is not None:
                suffix = up.name.lower().rsplit(".", 1)[-1]
                if suffix == "pdf":
                    raw_text = extract_pdf_pages_to_text(up, 1, 9999)
                elif suffix in {"txt", "md"}:
                    raw_text = up.read().decode("utf-8", errors="ignore")
                elif suffix == "json":
                    try:
                        uploaded_structured = json.load(up)
                    except Exception as e:
                        st.error(f"Failed to parse JSON file: {e}")

        model_choice = st.selectbox(
            "Analysis model",
            ["gemini-2.5-flash", "gemini-3-pro-preview"],
            index=0,
            key="k510_analysis_model",
            help="gemini-2.5-flash = faster; gemini-3-pro-preview = deeper reasoning",
        )

        max_tokens = st.number_input(
            "max_tokens for analysis",
            min_value=2000,
            max_value=32000,
            value=12000,
            step=1000,
            key="k510_analysis_max_tokens",
        )

        if st.button("Analyze 510(k) Summary", key="k510_analyze_btn"):
            if uploaded_structured is not None:
                st.session_state["k510_structured"] = uploaded_structured
                st.session_state["k510_raw_text"] = raw_text or ""
                st.session_state["k510_status"] = "done"
                st.success("Loaded structured 510(k) JSON from uploaded file.")
                token_est = int(len(json.dumps(uploaded_structured)) / 4)
                log_event(
                    "510(k) Summary Studio",
                    "510(k) JSON Import",
                    "local-json",
                    token_est,
                )
            else:
                if not raw_text.strip():
                    st.warning("Please paste or upload a 510(k) summary before analysis.")
                else:
                    st.session_state["k510_status"] = "running"
                    show_status("510(k) Summary Analysis", "running")
                    api_keys = st.session_state.get("api_keys", {})
                    with st.spinner("Parsing 510(k) summary with Gemini..."):
                        try:
                            parsed = parse_510k_summary_with_gemini(
                                text=raw_text,
                                model=model_choice,
                                api_keys=api_keys,
                                max_tokens=max_tokens,
                                temperature=0.1,
                            )
                            st.session_state["k510_structured"] = parsed
                            st.session_state["k510_raw_text"] = raw_text
                            st.session_state["k510_status"] = "done"
                            token_est = int(len(raw_text + json.dumps(parsed)) / 4)
                            log_event(
                                "510(k) Summary Studio",
                                "510(k) Summary Parser",
                                model_choice,
                                token_est,
                            )
                            st.success("510(k) summary parsed successfully.")
                        except Exception as e:
                            st.session_state["k510_status"] = "error"
                            st.error(f"Analysis error: {e}")

    with col_input_right:
        st.markdown("**Current status**")
        status = st.session_state.get("k510_status", "pending")
        if status == "done":
            st.success("Latest 510(k) summary successfully analyzed.")
        elif status == "running":
            st.info("Analysis in progress...")
        elif status == "error":
            st.error("The last analysis encountered an error. Adjust input and try again.")
        else:
            st.caption("No 510(k) summary analyzed yet in this session.")

        if "k510_structured" in st.session_state:
            st.markdown("**Quick JSON preview**")
            st.json(st.session_state["k510_structured"], expanded=False)

    structured = st.session_state.get("k510_structured")
    original_text = st.session_state.get("k510_raw_text", "")

    if structured:
        st.markdown("---")
        st.markdown("### 510(k) Summary Dashboard & Interactive QA")

        left, right = st.columns([3, 2], gap="large")
        with left:
            render_510k_summary_dashboard(structured)
        with right:
            render_510k_contextual_chat(structured, original_text)
    else:
        st.info("Once a 510(k) summary is analyzed, a visual dashboard and contextual Q&A panel will appear here.")


def render_pdf_to_md_tab():
    st.title("PDF & Office Transformer")

    # 1. DOCX/DOC → PDF + OCR (Gemini)
    st.markdown("### 1. DOCX/DOC → PDF with OCR (Gemini 2.5 Flash)")

    office_file = st.file_uploader(
        "Upload DOCX or DOC file",
        type=["docx", "doc"],
        key="office_to_pdf_uploader",
    )

    if office_file is not None:
        if st.button("Convert to PDF", key="convert_office_to_pdf_btn"):
            try:
                pdf_bytes, page_count = convert_office_to_pdf(office_file)
                st.session_state["ocr_pdf_bytes"] = pdf_bytes
                st.session_state["ocr_pdf_page_count"] = page_count
                st.success(f"Converted to PDF with {page_count} page(s).")
            except Exception as e:
                st.error(f"Conversion failed: {e}")

    pdf_bytes = st.session_state.get("ocr_pdf_bytes", None)
    page_count = st.session_state.get("ocr_pdf_page_count", None)

    if pdf_bytes:
        st.markdown("#### Generated PDF Preview")
        show_pdf(pdf_bytes, height=600)

        st.download_button(
            "Download generated PDF",
            data=pdf_bytes,
            file_name="converted_document.pdf",
            mime="application/pdf",
            key="download_converted_pdf",
        )

        if page_count is None:
            reader = PdfReader(BytesIO(pdf_bytes))
            page_count = len(reader.pages)
            st.session_state["ocr_pdf_page_count"] = page_count

        st.markdown("#### OCR on Selected Pages (Gemini 2.5 Flash)")

        col1, col2 = st.columns(2)
        with col1:
            from_page = st.number_input(
                "From page",
                min_value=1,
                max_value=page_count,
                value=1,
                key="ocr_from_page",
            )
        with col2:
            to_page = st.number_input(
                "To page",
                min_value=1,
                max_value=page_count,
                value=page_count,
                key="ocr_to_page",
            )

        if st.button("Run OCR (Gemini 2.5 Flash)", key="run_ocr_gemini_btn"):
            if from_page > to_page:
                st.error("`From page` must be <= `To page`.")
            else:
                try:
                    text_for_ocr = extract_pdf_pages_to_text(
                        BytesIO(pdf_bytes),
                        int(from_page),
                        int(to_page),
                    )
                    if not text_for_ocr.strip():
                        st.warning("No text extracted from the selected pages.")
                    else:
                        api_keys = st.session_state.get("api_keys", {})

                        system_prompt = f"""
You are an OCR post-processor using the Gemini 2.5 Flash model.

Input: noisy or unstructured text extracted from scanned or converted document pages.

Tasks:
1. Reconstruct the content as clean, well-structured markdown.
   - Preserve logical headings, lists, and tables (as markdown tables) where possible.
   - Fix obvious OCR errors (broken words, merged lines, split words).
2. Do NOT hallucinate new content. Only rewrite/clean what is present.
3. Identify important domain-specific keywords (technical terms, product names,
   standards, test types, risks, regulatory concepts).
4. Wrap each important keyword or short phrase in an HTML span with coral color and
   semi-bold weight:

   <span style="color:coral;font-weight:600;">keyword</span>

5. Output must be valid markdown with inline HTML spans.

Language: {st.session_state.settings["language"]}.
"""
                        user_prompt = text_for_ocr

                        with st.spinner("Running Gemini OCR to markdown..."):
                            out = call_llm(
                                model="gemini-2.5-flash",
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                max_tokens=st.session_state.settings["max_tokens"],
                                temperature=0.1,
                                api_keys=api_keys,
                            )

                        st.session_state["ocr_md_output"] = out
                        token_est = int(len(user_prompt + out) / 4)
                        log_event(
                            "PDF → OCR",
                            "Gemini-2.5-Flash OCR Markdown",
                            "gemini-2.5-flash",
                            token_est,
                        )
                        st.success("OCR complete.")

                except Exception as e:
                    st.error(f"OCR failed: {e}")

        ocr_md = st.session_state.get("ocr_md_output", "")
        if ocr_md:
            st.markdown("##### OCR Result (editable)")

            view_mode = st.radio(
                "View mode",
                ["Markdown", "Plain text"],
                horizontal=True,
                key="ocr_output_viewmode",
            )
            if view_mode == "Markdown":
                edited = st.text_area(
                    "OCR Markdown (editable)",
                    value=ocr_md,
                    height=320,
                    key="ocr_md_edited",
                )
            else:
                edited = st.text_area(
                    "OCR Text (editable)",
                    value=ocr_md,
                    height=320,
                    key="ocr_txt_edited",
                )

            st.session_state["ocr_effective_text"] = edited

    st.markdown("---")
    # 2. Classic PDF → Markdown Agent
    st.markdown("### 2. Classic PDF → Markdown Transformer (Agent)")

    uploaded = st.file_uploader(
        "Upload 510(k) or related PDF for direct PDF → Markdown",
        type=["pdf"],
        key="pdf_to_md_uploader",
    )
    if uploaded:
        col1, col2 = st.columns(2)
        with col1:
            num_start = st.number_input("From page", min_value=1, value=1, key="pdf_to_md_from")
        with col2:
            num_end = st.number_input("To page", min_value=1, value=10, key="pdf_to_md_to")

        if st.button("Extract Pages", key="pdf_to_md_extract_btn"):
            text = extract_pdf_pages_to_text(uploaded, int(num_start), int(num_end))
            st.session_state["pdf_raw_text"] = text

    raw_text = st.session_state.get("pdf_raw_text", "")
    if raw_text:
        default_prompt = f"""
You are converting part of a regulatory PDF into markdown.

- Input pages: a 510(k) submission, guidance, or related document excerpt.
- Goal: produce clean, structured markdown preserving headings, lists,
  and tables (as markdown tables) as much as reasonably possible.
- Do not hallucinate content that is not in the text.
- Clearly separate sections corresponding to major headings.

Language: {st.session_state.settings["language"]}.
"""
        agent_run_ui(
            agent_id="pdf_to_markdown_agent",
            tab_key="pdf_to_md",
            default_prompt=default_prompt,
            default_input_text=raw_text,
            tab_label_for_history="PDF → Markdown",
        )
    else:
        st.info("Upload a PDF and click 'Extract Pages' to begin, or use the DOCX/DOC → PDF workflow above.")


def render_summary_tab():
    st.title("Comprehensive Summary & Entities")

    base_md = st.session_state.get("pdf_to_md_output_edited", "")
    if base_md:
        default_input = base_md
    else:
        default_input = st.text_area("Paste markdown to summarize", value="", height=300)

    default_prompt = f"""
You are assisting an FDA 510(k) reviewer.

Given the following markdown document (derived from a 510(k) or related
submission), perform two tasks:

1. Produce a 3000–4000 word high-quality summary structured for a 510(k)
   review memo.
2. Extract at least 20 key entities and present them in a markdown table.

Language: {st.session_state.settings["language"]}.
"""

    agent_run_ui(
        agent_id="summary_entities_agent",
        tab_key="summary",
        default_prompt=default_prompt,
        default_input_text=default_input,
        tab_label_for_history="Summary & Entities",
    )


def render_diff_tab():
    st.title("Dual-Version Comparator")

    col1, col2 = st.columns(2)
    with col1:
        pdf_old = st.file_uploader("Upload Old Version PDF", type=["pdf"], key="pdf_old")
    with col2:
        pdf_new = st.file_uploader("Upload New Version PDF", type=["pdf"], key="pdf_new")

    if pdf_old and pdf_new and st.button("Extract Text for Comparison"):
        st.session_state["old_text"] = extract_pdf_pages_to_text(pdf_old, 1, 9999)
        st.session_state["new_text"] = extract_pdf_pages_to_text(pdf_new, 1, 9999)

    old_txt = st.session_state.get("old_text", "")
    new_txt = st.session_state.get("new_text", "")

    if old_txt and new_txt:
        combined = f"=== OLD VERSION ===\n{old_txt}\n\n=== NEW VERSION ===\n{new_txt}"

        default_prompt = f"""
You are comparing two versions of a 510(k)-related document.

Tasks:
1. Identify meaningful differences between the OLD and NEW versions.
2. Present them in a markdown table with regulatory-focused comments.

Language: {st.session_state.settings["language"]}.
"""

        agent_run_ui(
            agent_id="diff_agent",
            tab_key="diff",
            default_prompt=default_prompt,
            default_input_text=combined,
            tab_label_for_history="Comparator",
        )

        st.markdown("---")
        st.subheader("Run additional agents from agents.yaml on this combined doc")

        agents_cfg = st.session_state["agents_cfg"]
        agent_ids = list(agents_cfg["agents"].keys())
        selected_agents = st.multiselect(
            "Select agents to run on the current combined document",
            agent_ids,
        )

        current_text = st.session_state.get("diff_output_edited", combined)
        for aid in selected_agents:
            st.markdown(f"#### Agent: {agents_cfg['agents'][aid]['name']}")
            agent_run_ui(
                agent_id=aid,
                tab_key=f"diff_{aid}",
                default_prompt=agents_cfg["agents"][aid].get("system_prompt", ""),
                default_input_text=current_text,
                tab_label_for_history=f"Comparator-{aid}",
            )
            current_text = st.session_state.get(f"diff_{aid}_output_edited", current_text)
    else:
        st.info("Upload both old and new PDFs, then click 'Extract Text for Comparison'.")


def render_checklist_tab():
    st.title("Review Checklist & Report")

    st.subheader("Step 1: Provide Review Guidance")
    guidance_file = st.file_uploader("Upload guidance (PDF/MD/TXT)", type=["pdf", "md", "txt"])
    guidance_text = ""
    if guidance_file:
        if guidance_file.type == "application/pdf":
            guidance_text = extract_pdf_pages_to_text(guidance_file, 1, 9999)
        else:
            guidance_text = guidance_file.read().decode("utf-8", errors="ignore")

    manual_guidance = st.text_area("Or paste guidance text/markdown", height=200)
    guidance_text = guidance_text or manual_guidance

    if guidance_text:
        default_prompt = st.session_state["agents_cfg"]["agents"].get(
            "guidance_to_checklist_converter", {}
        ).get("user_prompt_template", f"""
Please generate a comprehensive review checklist based on the following FDA guidance:

{guidance_text}
""")

        if "guidance_to_checklist_converter" in st.session_state["agents_cfg"]["agents"]:
            agent_run_ui(
                agent_id="guidance_to_checklist_converter",
                tab_key="checklist",
                default_prompt=default_prompt,
                default_input_text=guidance_text,
                tab_label_for_history="Checklist",
            )
        else:
            st.warning("Agent 'guidance_to_checklist_converter' not found in agents.yaml.")

    st.markdown("---")
    st.subheader("Step 2: Build Review Report")

    checklist_md = st.session_state.get("checklist_output_edited", "")
    review_results_file = st.file_uploader("Upload review results (TXT/MD)", type=["txt", "md"])
    review_results_text = ""
    if review_results_file:
        review_results_text = review_results_file.read().decode("utf-8", errors="ignore")
    review_results_manual = st.text_area("Or paste review results", height=200)
    review_results = review_results_text or review_results_manual

    if checklist_md and review_results:
        default_prompt = st.session_state["agents_cfg"]["agents"].get(
            "review_memo_builder", {}
        ).get("user_prompt_template", f"""
Please compile the following checklist and review results into a formal FDA 510(k)
review memorandum.

=== CHECKLIST ===
{checklist_md}

=== REVIEW RESULTS ===
{review_results}
""")

        combined_input = f"=== CHECKLIST ===\n{checklist_md}\n\n=== REVIEW RESULTS ===\n{review_results}"

        if "review_memo_builder" in st.session_state["agents_cfg"]["agents"]:
            agent_run_ui(
                agent_id="review_memo_builder",
                tab_key="review_report",
                default_prompt=default_prompt,
                default_input_text=combined_input,
                tab_label_for_history="Review Report",
            )
        else:
            st.warning("Agent 'review_memo_builder' not found in agents.yaml.")
    else:
        st.info("Provide both a checklist and review results to generate a report.")


def render_note_keeper_tab():
    st.title("AI Note Keeper & Magics")

    raw_notes = st.text_area("Paste your notes (text or markdown)", height=300, key="notes_raw")
    if raw_notes:
        default_prompt = """
You are restructuring a 510(k) reviewer's notes into organized markdown.

Tasks:
1. Identify major sections and sub-sections.
2. Convert bullet fragments into readable sentences where helpful.
3. Highlight key points, open questions, and follow-up items.
4. Avoid inventing information not present in the notes.
"""
        agent_run_ui(
            agent_id="note_keeper_agent",
            tab_key="notes",
            default_prompt=default_prompt,
            default_input_text=raw_notes,
            tab_label_for_history="Note Keeper",
        )

    processed = st.session_state.get("notes_output_edited", raw_notes)

    st.markdown("---")
    st.subheader("AI Magics")

    st.markdown("Select a Magic and apply it to the current note.")
    magic_options = {
        "AI Formatting": "magic_formatting_agent",
        "AI Keywords": "magic_keywords_agent",
        "AI Action Items": "magic_action_items_agent",
        "AI Concept Map": "magic_concept_map_agent",
        "AI Glossary": "magic_glossary_agent",
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        magic_name = st.selectbox("Magic", list(magic_options.keys()))
    with col2:
        keyword_color = st.color_picker("Keyword color (for AI Keywords)", "#ff7f50")  # coral default

    if st.button("Apply Magic"):
        agent_id = magic_options[magic_name]
        base_prompt = st.session_state["agents_cfg"]["agents"][agent_id]["system_prompt"]
        if magic_name == "AI Keywords":
            magic_prompt_suffix = f"""
When returning keywords, identify the most important regulatory and technical
keywords. Wrap each keyword in an HTML span with inline style using this color:
{keyword_color}.

Example:
- <span style="color:{keyword_color};font-weight:bold;">predicate device</span>
"""
        else:
            magic_prompt_suffix = ""

        full_prompt = base_prompt + "\n\n" + magic_prompt_suffix

        agent_run_ui(
            agent_id=agent_id,
            tab_key=f"magic_{agent_id}",
            default_prompt=full_prompt,
            default_input_text=processed,
            tab_label_for_history=f"Magic-{magic_name}",
        )


def render_fda_orchestration_tab():
    st.title("FDA Reviewer Orchestration")

    # Step 1 – Provide Device Description
    st.subheader("Step 1 – Provide Device Description (text / markdown / JSON or file upload)")

    col1, col2 = st.columns(2)
    with col1:
        device_raw_text = st.text_area(
            "Paste device description (device name, classification, description, intended use, etc.)",
            height=260,
            key="orch_device_raw",
        )
    with col2:
        file = st.file_uploader("Or upload device file (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
        if file is not None:
            if file.type == "application/pdf":
                extracted = extract_pdf_pages_to_text(file, 1, 9999)
            elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                extracted = extract_docx_to_text(file)
            else:  # txt
                extracted = file.read().decode("utf-8", errors="ignore")
            if st.button("Use uploaded file as device description"):
                st.session_state["orch_device_raw"] = extracted
                device_raw_text = extracted
                st.success("Loaded device description from file.")

    # Step 1b – Structure with coral keywords
    st.markdown("#### Step 1b – Transform into organized markdown with highlighted keywords")
    if "orch_device_struct_status" not in st.session_state:
        st.session_state["orch_device_struct_status"] = "pending"
    show_status("Device Description Structuring", st.session_state["orch_device_struct_status"])

    colm1, colm2 = st.columns([2, 1])
    with colm1:
        struct_model = st.selectbox(
            "Model for structuring",
            ALL_MODELS,
            index=ALL_MODELS.index(st.session_state.settings["model"]),
            key="orch_device_model",
        )
    with colm2:
        struct_max_tokens = st.number_input(
            "max_tokens",
            min_value=1000,
            max_value=120000,
            value=st.session_state.settings["max_tokens"],
            step=1000,
            key="orch_device_max_tokens",
        )

    if st.button("Transform to structured markdown", key="orch_device_run"):
        if not device_raw_text.strip():
            st.warning("Please provide device text or upload a file first.")
        else:
            st.session_state["orch_device_struct_status"] = "running"
            show_status("Device Description Structuring", "running")
            api_keys = st.session_state.get("api_keys", {})
            system_prompt = """
You are a medical device documentation organizer.

Input: unstructured device description text.

Task:
1. Produce a CLEAN, well-structured markdown device description with sections.
2. Highlight important regulatory and technical keywords using:
   <span style="color:coral;font-weight:600;">keyword</span>
3. Do NOT invent new facts.
"""
            user_prompt = device_raw_text

            with st.spinner("Transforming device description..."):
                try:
                    out = call_llm(
                        model=struct_model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=struct_max_tokens,
                        temperature=0.15,
                        api_keys=api_keys,
                    )
                    st.session_state["orch_device_md"] = out
                    st.session_state["orch_device_struct_status"] = "done"
                    token_est = int(len(user_prompt + out) / 4)
                    log_event(
                        "FDA Orchestration",
                        "Device Description Structurer",
                        struct_model,
                        token_est,
                    )
                except Exception as e:
                    st.session_state["orch_device_struct_status"] = "error"
                    st.error(f"Error structuring device description: {e}")

    st.markdown("##### Structured Device Description (editable)")
    device_md = st.session_state.get("orch_device_md", device_raw_text)
    view_mode = st.radio(
        "View mode", ["Markdown", "Plain text"],
        horizontal=True, key="orch_device_viewmode"
    )
    if view_mode == "Markdown":
        device_md_edited = st.text_area(
            "Device description (Markdown)",
            value=device_md,
            height=260,
            key="orch_device_md_edited",
        )
    else:
        device_md_edited = st.text_area(
            "Device description (Plain text)",
            value=device_md,
            height=260,
            key="orch_device_txt_edited",
        )
    st.session_state["orch_device_md_effective"] = device_md_edited

    # Step 2 – Agents Catalog overview
    st.markdown("---")
    st.subheader("Step 2 – Select agents.yaml (default or uploaded)")

    agents_cfg = st.session_state["agents_cfg"]
    with st.expander("Current agents catalog overview"):
        agents_df = pd.DataFrame([
            {
                "agent_id": aid,
                "name": acfg.get("name", ""),
                "category": acfg.get("category", ""),
            }
            for aid, acfg in agents_cfg.get("agents", {}).items()
        ])
        st.dataframe(agents_df, use_container_width=True, height=240)

    # Step 3 – Orchestration
    st.markdown("---")
    st.subheader("Step 3 – Generate FDA 510(k) Review Orchestration Plan")

    colx1, colx2, colx3 = st.columns(3)
    with colx1:
        submission_type = st.text_input("Submission Type (e.g., Traditional 510(k))", "")
    with colx2:
        regulatory_pathway = st.text_input("Regulatory Pathway", "510(k)")
    with colx3:
        clinical_data_status = st.selectbox(
            "Clinical Data Available?",
            ["Unclear", "Yes", "No"],
            index=0,
        )

    known_predicates = st.text_input("Known predicate devices (free text)", "")
    special_circumstances = st.text_area("Special circumstances (pediatric, home use, AI/ML, etc.)", height=80)

    depth_choice = st.radio(
        "Requested analysis depth",
        ["Quick Assessment", "Standard Orchestration", "Comprehensive Planning"],
        index=1,
        horizontal=True,
    )
    depth_quick = "☑ Quick Assessment (identify primary agents only)" if depth_choice == "Quick Assessment" \
        else "☐ Quick Assessment (identify primary agents only)"
    depth_standard = "☑ Standard Orchestration (full phase-based plan)" if depth_choice == "Standard Orchestration" \
        else "☐ Standard Orchestration (full phase-based plan)"
    depth_comprehensive = "☑ Comprehensive Planning (include timeline, challenges, execution commands)" \
        if depth_choice == "Comprehensive Planning" \
        else "☐ Comprehensive Planning (include timeline, challenges, execution commands)"

    base_user_prompt = FDA_ORCH_USER_TEMPLATE.format(
        device_information=device_md_edited or "[Device description not provided]",
        submission_type=submission_type or "[Not provided]",
        regulatory_pathway=regulatory_pathway or "[Not provided]",
        known_predicates=known_predicates or "[Not provided]",
        clinical_data_status=clinical_data_status,
        special_circumstances=special_circumstances or "[None noted]",
        depth_quick=depth_quick,
        depth_standard=depth_standard,
        depth_comprehensive=depth_comprehensive,
    )

    st.markdown("##### Orchestration Prompt (editable before running)")
    orch_prompt_text = st.text_area(
        "Orchestration prompt",
        value=st.session_state.get("orch_prompt_text", base_user_prompt),
        height=260,
        key="orch_prompt_text",
    )

    colp1, colp2, colp3 = st.columns([2, 1, 1])
    with colp1:
        orch_model = st.selectbox(
            "Model for orchestration",
            ALL_MODELS,
            index=ALL_MODELS.index(st.session_state.settings["model"]),
            key="orch_model",
        )
    with colp2:
        orch_max_tokens = st.number_input(
            "max_tokens",
            min_value=8000,
            max_value=120000,
            value=max(20000, st.session_state.settings["max_tokens"]),
            step=2000,
            key="orch_max_tokens",
        )
    with colp3:
        if "orch_status" not in st.session_state:
            st.session_state["orch_status"] = "pending"
        show_status("FDA Review Orchestrator", st.session_state["orch_status"])

    if st.button("Run Orchestrator", key="orch_run"):
        st.session_state["orch_status"] = "running"
        show_status("FDA Review Orchestrator", "running")
        api_keys = st.session_state.get("api_keys", {})

        agents_yaml_str = yaml.dump(agents_cfg.get("agents", {}), allow_unicode=True, sort_keys=False)
        user_prompt_full = orch_prompt_text + "\n\n---\n\nAGENT CATALOG (agents.yaml excerpt):\n\n" + agents_yaml_str

        with st.spinner("Creating orchestration plan..."):
            try:
                out = call_llm(
                    model=orch_model,
                    system_prompt=FDA_ORCH_SYSTEM_PROMPT,
                    user_prompt=user_prompt_full,
                    max_tokens=orch_max_tokens,
                    temperature=0.2,
                    api_keys=api_keys,
                )
                st.session_state["orch_plan"] = out
                st.session_state["orch_status"] = "done"
                token_est = int(len(user_prompt_full + out) / 4)
                log_event(
                    "FDA Orchestration",
                    "FDA Review Orchestrator",
                    orch_model,
                    token_est,
                )
            except Exception as e:
                st.session_state["orch_status"] = "error"
                st.error(f"Error running orchestrator: {e}")

    st.markdown("##### Orchestration Plan (Markdown / text, editable)")
    orch_plan = st.session_state.get("orch_plan", "")
    view_mode2 = st.radio(
        "Plan view mode", ["Markdown", "Plain text"],
        horizontal=True, key="orch_plan_viewmode"
    )
    if view_mode2 == "Markdown":
        orch_plan_edited = st.text_area(
            "Orchestration Plan (Markdown)",
            value=orch_plan,
            height=360,
            key="orch_plan_md_edited",
        )
    else:
        orch_plan_edited = st.text_area(
            "Orchestration Plan (Plain text)",
            value=orch_plan,
            height=360,
            key="orch_plan_txt_edited",
        )
    st.session_state["orch_plan_effective"] = orch_plan_edited

    # Step 4 – Execute agents sequentially
    st.markdown("---")
    st.subheader("Step 4 – Execute Review Agents (chain outputs between agents)")

    agents_cfg = st.session_state["agents_cfg"]
    agent_ids = list(agents_cfg["agents"].keys())
    selected_agents = st.multiselect(
        "Select agents to run sequentially",
        agent_ids,
        help="You can run agents one by one using the orchestration plan or device description as input.",
    )

    base_chain_input = st.session_state.get("orch_plan_effective", "") or st.session_state.get(
        "orch_device_md_effective", ""
    )
    current_text = base_chain_input

    if not base_chain_input:
        st.info("Once you have an orchestration plan or structured device description, you can use it as the starting input here.")

    for aid in selected_agents:
        st.markdown(f"#### Agent: {agents_cfg['agents'][aid].get('name', aid)}")
        agent_run_ui(
            agent_id=aid,
            tab_key=f"orch_exec_{aid}",
            default_prompt=agents_cfg["agents"][aid].get("user_prompt_template", agents_cfg["agents"][aid].get("system_prompt", "")),
            default_input_text=current_text,
            tab_label_for_history=f"Orchestration-{aid}",
        )
        current_text = st.session_state.get(f"orch_exec_{aid}_output_edited", current_text)


def render_dynamic_agents_tab():
    st.title("Dynamic Review Agent Generator from FDA Guidance")

    # Step 1: Guidance ingestion
    st.subheader("Step 1 – Provide FDA Guidance Text")
    gfile = st.file_uploader("Upload guidance (PDF / MD / TXT)", type=["pdf", "md", "txt"], key="dyn_guidance_file")
    guidance_text = ""
    if gfile is not None:
        if gfile.type == "application/pdf":
            guidance_text = extract_pdf_pages_to_text(gfile, 1, 9999)
        else:
            guidance_text = gfile.read().decode("utf-8", errors="ignore")

    guidance_text_manual = st.text_area("Or paste guidance text/markdown here", height=220, key="dyn_guidance_manual")
    guidance_text = guidance_text or guidance_text_manual

    if not guidance_text.strip():
        st.info("Provide FDA guidance text to enable checklist and dynamic agent generation.")
        return

    # Step 2 – Optional checklist
    st.markdown("---")
    st.subheader("Step 2 – (Optional) Generate Review Checklist from Guidance")

    if "guidance_to_checklist_converter" in st.session_state["agents_cfg"]["agents"]:
        default_prompt = st.session_state["agents_cfg"]["agents"]["guidance_to_checklist_converter"].get(
            "user_prompt_template",
            f"Please generate a detailed review checklist from the following guidance:\n\n{guidance_text}\n",
        )
        agent_run_ui(
            agent_id="guidance_to_checklist_converter",
            tab_key="dyn_checklist",
            default_prompt=default_prompt,
            default_input_text=guidance_text,
            tab_label_for_history="Dynamic-Checklist",
        )
    else:
        st.warning("Agent 'guidance_to_checklist_converter' not found in agents.yaml; skipping checklist step.")

    checklist_md = st.session_state.get("dyn_checklist_output_edited", "")

    # Step 3 – Dynamic Agent Generator
    st.markdown("---")
    st.subheader("Step 3 – Generate New Specialized Agents from Guidance")

    if "dyn_agent_status" not in st.session_state:
        st.session_state["dyn_agent_status"] = "pending"
    show_status("Dynamic Agent Generator", st.session_state["dyn_agent_status"])

    col1, col2, col3 = st.columns(3)
    with col1:
        dyn_model = st.selectbox(
            "Model for dynamic agents",
            ALL_MODELS,
            index=ALL_MODELS.index(st.session_state.settings["model"]),
            key="dyn_agent_model",
        )
    with col2:
        dyn_max_tokens = st.number_input(
            "max_tokens",
            min_value=8000,
            max_value=120000,
            value=max(20000, st.session_state.settings["max_tokens"]),
            step=2000,
            key="dyn_agent_max_tokens",
        )
    with col3:
        num_agents_hint = st.slider(
            "Target number of new agents (hint to model)",
            min_value=3,
            max_value=8,
            value=5,
            step=1,
            key="dyn_agent_count",
        )

    current_agents_yaml = yaml.dump(
        st.session_state["agents_cfg"].get("agents", {}),
        allow_unicode=True,
        sort_keys=False,
    )
    dyn_user_prompt = f"""
You are given the following FDA guidance text:

=== FDA GUIDANCE TEXT ===
{guidance_text}

=== OPTIONAL CHECKLIST (if present) ===
{checklist_md if checklist_md else "[No checklist provided]"} 

=== EXISTING AGENTS CATALOG (agents.yaml excerpt) ===
{current_agents_yaml}

Your task: create approximately {num_agents_hint} NEW specialized review agents
(to be added to agents.yaml) that are carefully tailored to this guidance, without
duplicating existing agents.
"""

    if st.button("Generate New Agents.yaml Snippet", key="dyn_agent_run"):
        st.session_state["dyn_agent_status"] = "running"
        show_status("Dynamic Agent Generator", "running")
        api_keys = st.session_state.get("api_keys", {})

        with st.spinner("Generating new agent definitions from guidance..."):
            try:
                out = call_llm(
                    model=dyn_model,
                    system_prompt=DYNAMIC_AGENT_SYSTEM_PROMPT,
                    user_prompt=dyn_user_prompt,
                    max_tokens=dyn_max_tokens,
                    temperature=0.2,
                    api_keys=api_keys,
                )
                st.session_state["dyn_agent_yaml"] = out
                st.session_state["dyn_agent_status"] = "done"
                token_est = int(len(dyn_user_prompt + out) / 4)
                log_event(
                    "Dynamic Agents",
                    "Dynamic Agent Generator",
                    dyn_model,
                    token_est,
                )
            except Exception as e:
                st.session_state["dyn_agent_status"] = "error"
                st.error(f"Error generating dynamic agents: {e}")

    st.markdown("##### Generated agents.yaml snippet (editable)")
    dyn_yaml = st.session_state.get("dyn_agent_yaml", "")
    dyn_yaml_edited = st.text_area(
        "agents.yaml snippet",
        value=dyn_yaml,
        height=360,
        key="dyn_agent_yaml_edited",
    )

    if dyn_yaml_edited.strip():
        st.download_button(
            "Download agents.yaml snippet",
            data=dyn_yaml_edited.encode("utf-8"),
            file_name="dynamic_agents.yaml",
            mime="text/yaml",
        )
        st.info("You can merge this YAML into your main agents.yaml and reload the app (or upload via sidebar).")


def render_agents_config_tab():
    st.title("Agents Config Studio / 代理設定工作室")

    agents_cfg = st.session_state["agents_cfg"]
    agents_dict = agents_cfg.get("agents", {})

    # ---- Section 1: Overview ----
    st.subheader("1. Current Agents Overview")
    if not agents_dict:
        st.warning("No agents found in current agents.yaml.")
    else:
        df = pd.DataFrame([
            {
                "agent_id": aid,
                "name": acfg.get("name", ""),
                "model": acfg.get("model", ""),
                "category": acfg.get("category", ""),
            }
            for aid, acfg in agents_dict.items()
        ])
        st.dataframe(df, use_container_width=True, height=260)

    # ---- Section 2: Edit agents.yaml as raw text ----
    st.markdown("---")
    st.subheader("2. Edit Full agents.yaml (raw text)")

    yaml_str_current = yaml.dump(
        st.session_state["agents_cfg"],
        allow_unicode=True,
        sort_keys=False,
    )

    edited_yaml_text = st.text_area(
        "agents.yaml (editable)",
        value=yaml_str_current,
        height=320,
        key="agents_yaml_text_editor",
    )

    col_yaml_btn1, col_yaml_btn2, col_yaml_btn3 = st.columns(3)
    with col_yaml_btn1:
        if st.button("Apply edited YAML to session", key="apply_edited_yaml"):
            try:
                cfg = yaml.safe_load(edited_yaml_text)
                if not isinstance(cfg, dict) or "agents" not in cfg:
                    st.error("Parsed YAML does not contain top-level key 'agents'. No changes applied.")
                else:
                    st.session_state["agents_cfg"] = cfg
                    st.success("Updated agents.yaml in current session.")
            except Exception as e:
                st.error(f"Failed to parse edited YAML: {e}")

    with col_yaml_btn2:
        uploaded_agents_tab = st.file_uploader(
            "Upload agents.yaml file",
            type=["yaml", "yml"],
            key="agents_yaml_tab_uploader",
        )
        if uploaded_agents_tab is not None:
            try:
                cfg = yaml.safe_load(uploaded_agents_tab.read())
                if "agents" in cfg:
                    st.session_state["agents_cfg"] = cfg
                    st.success("Uploaded agents.yaml applied to this session.")
                else:
                    st.warning("Uploaded file has no top-level 'agents' key. Ignoring.")
            except Exception as e:
                st.error(f"Failed to parse uploaded YAML: {e}")

    with col_yaml_btn3:
        st.download_button(
            "Download current agents.yaml",
            data=yaml_str_current.encode("utf-8"),
            file_name="agents.yaml",
            mime="text/yaml",
            key="download_agents_yaml_current",
        )

    # ---- Section 3: Per-agent editor (model + prompts) ----
    st.markdown("---")
    st.subheader("3. Per-Agent Editor (Model & Prompts)")

    if not agents_dict:
        st.info("No agents to edit. Load an agents.yaml first.")
    else:
        agent_ids = list(agents_dict.keys())
        selected_agent_id = st.selectbox(
            "Select agent to edit",
            agent_ids,
            key="agents_config_selected_agent",
        )
        if selected_agent_id:
            agent_cfg = agents_dict[selected_agent_id]

            st.markdown(f"**Editing agent_id:** `{selected_agent_id}`")

            col_info1, col_info2 = st.columns([2, 1])
            with col_info1:
                name = st.text_input("Name", value=agent_cfg.get("name", ""), key="agent_edit_name")
                category = st.text_input("Category", value=agent_cfg.get("category", ""), key="agent_edit_category")
            with col_info2:
                description = st.text_area("Description", value=agent_cfg.get("description", ""), height=80, key="agent_edit_description")

            col_model, col_temp, col_maxtok = st.columns(3)
            with col_model:
                current_model = agent_cfg.get("model", AGENT_MODEL_CHOICES[0])
                if current_model not in AGENT_MODEL_CHOICES:
                    current_model = AGENT_MODEL_CHOICES[0]
                model_choice = st.selectbox(
                    "Model (for this agent)",
                    AGENT_MODEL_CHOICES,
                    index=AGENT_MODEL_CHOICES.index(current_model),
                    key="agent_edit_model",
                )
            with col_temp:
                temperature = st.number_input(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(agent_cfg.get("temperature", 0.2)),
                    step=0.05,
                    key="agent_edit_temperature",
                )
            with col_maxtok:
                max_tokens = st.number_input(
                    "max_tokens",
                    min_value=1000,
                    max_value=120000,
                    value=int(agent_cfg.get("max_tokens", 12000)),
                    step=1000,
                    key="agent_edit_max_tokens",
                )

            st.markdown("**system_prompt**")
            system_prompt = st.text_area(
                "system_prompt",
                value=agent_cfg.get("system_prompt", ""),
                height=200,
                key="agent_edit_system_prompt",
            )

            st.markdown("**user_prompt_template**")
            user_prompt_template = st.text_area(
                "user_prompt_template",
                value=agent_cfg.get("user_prompt_template", ""),
                height=200,
                key="agent_edit_user_prompt_template",
            )

            if st.button("Save changes to this agent", key="save_agent_changes_btn"):
                try:
                    updated = agent_cfg.copy()
                    updated["name"] = name
                    updated["category"] = category
                    updated["description"] = description
                    updated["model"] = model_choice
                    updated["temperature"] = float(temperature)
                    updated["max_tokens"] = int(max_tokens)
                    updated["system_prompt"] = system_prompt
                    updated["user_prompt_template"] = user_prompt_template

                    st.session_state["agents_cfg"]["agents"][selected_agent_id] = updated
                    st.success(f"Agent '{selected_agent_id}' updated in session.")
                except Exception as e:
                    st.error(f"Failed to update agent: {e}")

    # ---- Section 4: Standardize agents.yaml via LLM ----
    st.markdown("---")
    st.subheader("4. Standardize agents.yaml Text via LLM")

    st.markdown(
        "Paste any agents-related YAML or free-form description below. "
        "Optionally provide an example of your desired standardized agents.yaml format. "
        "The system will transform it into a normalized agents.yaml with top-level `agents:`."
    )

    raw_agents_text = st.text_area(
        "Raw agents text (YAML or semi-structured)",
        value="",
        height=220,
        key="raw_agents_text_to_standardize",
    )

    example_format_text = st.text_area(
        "Optional: Example of desired standardized agents.yaml format",
        value="",
        height=200,
        key="example_agents_format_text",
    )

    col_std1, col_std2 = st.columns([2, 1])
    with col_std1:
        std_model = st.selectbox(
            "Model for standardization",
            AGENT_MODEL_CHOICES,
            index=AGENT_MODEL_CHOICES.index("gemini-2.5-flash"),
            key="standardize_agents_model",
        )
    with col_std2:
        std_max_tokens = st.number_input(
            "max_tokens",
            min_value=4000,
            max_value=120000,
            value=12000,
            step=1000,
            key="standardize_agents_max_tokens",
        )

    if st.button("Standardize agents.yaml via LLM", key="run_standardize_agents_btn"):
        if not raw_agents_text.strip():
            st.warning("Please paste some agents text to standardize.")
        else:
            api_keys = st.session_state.get("api_keys", {})
            user_prompt_parts = [
                "=== RAW AGENTS TEXT ===",
                raw_agents_text,
            ]
            if example_format_text.strip():
                user_prompt_parts.extend([
                    "",
                    "=== EXAMPLE OF DESIRED STANDARD FORMAT (REFERENCE) ===",
                    example_format_text,
                ])
            user_prompt = "\n".join(user_prompt_parts)

            with st.spinner("Standardizing agents.yaml via LLM..."):
                try:
                    out = call_llm(
                        model=std_model,
                        system_prompt=STANDARDIZE_AGENTS_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        max_tokens=std_max_tokens,
                        temperature=0.15,
                        api_keys=api_keys,
                    )
                    st.session_state["standardized_agents_yaml"] = out
                    token_est = int(len(user_prompt + out) / 4)
                    log_event(
                        "Agents Config",
                        "Agents.yaml Standardizer",
                        std_model,
                        token_est,
                    )
                    st.success("Standardization completed. Review the result below.")
                except Exception as e:
                    st.error(f"Standardization failed: {e}")

    std_yaml = st.session_state.get("standardized_agents_yaml", "")
    if std_yaml:
        st.markdown("##### Standardized agents.yaml (editable)")
        std_yaml_edited = st.text_area(
            "Standardized agents.yaml",
            value=std_yaml,
            height=320,
            key="standardized_agents_yaml_edited",
        )

        col_apply_std1, col_apply_std2 = st.columns(2)
        with col_apply_std1:
            if st.button("Load standardized YAML into current session", key="load_standardized_yaml_btn"):
                try:
                    cfg = yaml.safe_load(std_yaml_edited)
                    if not isinstance(cfg, dict) or "agents" not in cfg:
                        st.error("Standardized YAML does not contain top-level 'agents'. Not applied.")
                    else:
                        st.session_state["agents_cfg"] = cfg
                        st.success("Standardized agents.yaml loaded into current session.")
                except Exception as e:
                    st.error(f"Failed to parse standardized YAML: {e}")

        with col_apply_std2:
            st.download_button(
                "Download standardized agents.yaml",
                data=std_yaml_edited.encode("utf-8"),
                file_name="standardized_agents.yaml",
                mime="text/yaml",
                key="download_standardized_agents_yaml",
            )


# =========================
# Main app
# =========================

st.set_page_config(page_title="FDA 510(k) Agentic Reviewer", layout="wide")

# Initialize session state
if "settings" not in st.session_state:
    st.session_state["settings"] = {
        "theme": "Light",
        "language": "English",
        "painter_style": "Van Gogh",
        "model": "gpt-4o-mini",
        "max_tokens": 12000,
        "temperature": 0.2,
    }

if "history" not in st.session_state:
    st.session_state["history"] = []

# Load default agents.yaml once (can be overridden in sidebar or Agents Config tab)
if "agents_cfg" not in st.session_state:
    try:
        with open("agents.yaml", "r", encoding="utf-8") as f:
            st.session_state["agents_cfg"] = yaml.safe_load(f)
    except Exception as e:
        st.error(f"Failed to load agents.yaml: {e}")
        st.stop()

# Render sidebar
render_sidebar()

# Apply WOW painter style & theme CSS
apply_style(st.session_state.settings["theme"], st.session_state.settings["painter_style"])

# Tabs with localized labels + new 510(k) Summary Studio + Agents Config tab
lang = st.session_state.settings["language"]
tab_labels = [
    t("Dashboard"),
    t("510k_tab"),
    t("510k_summary_studio"),
    t("PDF → Markdown"),
    t("Summary & Entities"),
    t("Comparator"),
    t("Checklist & Report"),
    t("Note Keeper & Magics"),
    t("FDA Orchestration"),
    t("Dynamic Agents"),
    t("Agents Config"),
]
tabs = st.tabs(tab_labels)

with tabs[0]:
    render_dashboard()
with tabs[1]:
    render_510k_tab()
with tabs[2]:
    render_510k_summary_studio_tab()
with tabs[3]:
    render_pdf_to_md_tab()
with tabs[4]:
    render_summary_tab()
with tabs[5]:
    render_diff_tab()
with tabs[6]:
    render_checklist_tab()
with tabs[7]:
    render_note_keeper_tab()
with tabs[8]:
    render_fda_orchestration_tab()
with tabs[9]:
    render_dynamic_agents_tab()
with tabs[10]:
    render_agents_config_tab()
