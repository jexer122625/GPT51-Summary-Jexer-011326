import os
from datetime import datetime
from io import BytesIO
import base64
import json  # NEW
```

---

## 2. Models & LABELS: add `gemini-3-pro-preview` and new tab label

### 2.1 Update `ALL_MODELS` and `GEMINI_MODELS`

Replace your `ALL_MODELS` and `GEMINI_MODELS` definitions with:

```python
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
    "gemini-3-pro-preview",       # NEW
}
ANTHROPIC_MODELS = {
    "claude-3-5-sonnet-20210",
    "claude-3-5-sonnet-2024-10",
    "claude-3-5-haiku-20241022",
}
GROK_MODELS = {"grok-4-fast-reasoning", "grok-3-mini"}
```

### 2.2 Optionally: allow agents to pick `gemini-3-pro-preview`

If you want agents in the Agents Config Studio to be able to use it, update `AGENT_MODEL_CHOICES`:

```python
AGENT_MODEL_CHOICES = [
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",   # NEW
    "gpt-4o-mini",
    "gpt-5-mini",
]
```

### 2.3 Add a localized label for the new tab

Extend your `LABELS` dict:

```python
LABELS = {
    "Dashboard": {"English": "Dashboard", "繁體中文": "儀表板"},
    "510k_tab": {"English": "510(k) Intelligence", "繁體中文": "510(k) 智能分析"},
    "510k_summary_studio": {  # NEW
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
```

---

## 3. New constants & helper functions for 510(k) Summary Intelligence

Add the following **after** your existing helper functions (e.g., after `agent_run_ui`), before the sidebar renderer.

### 3.1 510(k) JSON “schema” (prompt-driven)

This is a structured JSON layout that the Gemini model will target.  
We rely on `response_mime_type="application/json"` to strongly bias JSON output.

```python
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
```

### 3.2 Gemini-based parser for 510(k) summaries

```python
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

    # We bias the model toward returning valid JSON using response_mime_type.
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
        # Fallback: try to extract JSON substring
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
```

### 3.3 Dashboard renderer for structured 510(k) JSON

```python
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
        perf_tables = []

        # Build a flat table for infographics
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

        # Standards table
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
```

### 3.4 Contextual chat panel for the analyzed document

This uses the structured JSON plus original text as context and allows the user to choose `gemini-2.5-flash` (fast) or `gemini-3-pro-preview` (deeper).

```python
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

        # Build a simple linear conversation string (we keep it short for safety)
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
```

---

## 4. New tab renderer: “510(k) Summary Studio”

Add this **after** `render_510k_tab()` (or anywhere among tab renderers).

```python
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
                # Already structured JSON provided by user
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

    # --- Dashboard + Chat, once we have structured data ---
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
```

---

## 5. Wire the new tab into the main layout

At the bottom of your script, where you define `tab_labels` and `tabs`, insert the new tab **after** the existing 510(k) Intelligence tab.

### 5.1 Update `tab_labels`

Replace:

```python
tab_labels = [
    t("Dashboard"),
    t("510k_tab"),
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
    render_pdf_to_md_tab()
with tabs[3]:
    render_summary_tab()
with tabs[4]:
    render_diff_tab()
with tabs[5]:
    render_checklist_tab()
with tabs[6]:
    render_note_keeper_tab()
with tabs[7]:
    render_fda_orchestration_tab()
with tabs[8]:
    render_dynamic_agents_tab()
with tabs[9]:
    render_agents_config_tab()
```

with this (note the extra tab at index 2 and the shifted indices):

```python
tab_labels = [
    t("Dashboard"),
    t("510k_tab"),
    t("510k_summary_studio"),   # NEW
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
    render_510k_summary_studio_tab()   # NEW TAB
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
