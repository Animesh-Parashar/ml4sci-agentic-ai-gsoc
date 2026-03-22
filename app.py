import os
import asyncio
import base64
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

from src.config import AgentDependencies, SUPPORTED_MODELS, KeyRotator
from src.agent import create_lensing_agent

st.set_page_config(
    page_title="DeepLense Simulator",
    page_icon="🌌",
    layout="wide",
)


ASSISTANT_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" width="40" height="40">
  <defs>
    <radialGradient id="abg" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#241c14"/>
      <stop offset="100%" stop-color="#0e0c0a"/>
    </radialGradient>
  </defs>
  <circle cx="20" cy="20" r="20" fill="url(#abg)"/>
  <ellipse cx="20" cy="20" rx="15" ry="6" fill="none" stroke="#c49040" stroke-width="1" stroke-opacity="0.55" transform="rotate(-18 20 20)">
    <animate attributeName="stroke-opacity" values="0.35;0.7;0.35" dur="3s" repeatCount="indefinite"/>
    <animate attributeName="rx" values="15;16;15" dur="3s" repeatCount="indefinite"/>
  </ellipse>
  <ellipse cx="20" cy="20" rx="9" ry="3.5" fill="none" stroke="#e0b060" stroke-width="1.4" transform="rotate(-18 20 20)">
    <animate attributeName="stroke-opacity" values="0.7;1;0.7" dur="2.2s" repeatCount="indefinite"/>
  </ellipse>
  <circle cx="20" cy="20" r="2.5" fill="#f0cc80">
    <animate attributeName="opacity" values="0.7;1;0.7" dur="1.8s" repeatCount="indefinite"/>
  </circle>
</svg>"""

USER_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" width="40" height="40">
  <defs>
    <radialGradient id="ubg" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#0f1e1e"/><stop offset="100%" stop-color="#080e0e"/>
    </radialGradient>
    <linearGradient id="ufill" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#5bbcbc"/><stop offset="100%" stop-color="#257878"/>
    </linearGradient>
  </defs>
  <circle cx="20" cy="20" r="20" fill="url(#ubg)"/>
  <polygon points="20,8 30,14 30,26 20,32 10,26 10,14" fill="url(#ufill)" opacity="0.82">
    <animate attributeName="opacity" values="0.72;0.92;0.72" dur="2.8s" repeatCount="indefinite"/>
  </polygon>
  <polygon points="20,13 26,17 26,23 20,27 14,23 14,17" fill="#9ee8e8" opacity="0.55">
    <animate attributeName="opacity" values="0.4;0.7;0.4" dur="2s" repeatCount="indefinite"/>
  </polygon>
</svg>"""

def svg_to_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.strip().encode()).decode()

ASSISTANT_AVATAR = svg_to_uri(ASSISTANT_SVG)
USER_AVATAR      = svg_to_uri(USER_SVG)

# Global CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* ── Reset all Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
html, body, .stApp, .stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"] {
    background: #0f0e0c !important;
    font-family: 'Inter', system-ui, sans-serif;
    color: #c8c0b4;
}

/* ── Main container: full height, centered narrow column ── */
.main .block-container {
    max-width: 780px !important;
    margin: 0 auto !important;
    padding: 0 1rem 100px !important;
    min-height: 100vh;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2620; border-radius: 4px; }

/* ── Top nav bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 0.8rem;
    border-bottom: 1px solid #1c1a16;
    margin-bottom: 0;
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1.35rem;
    font-weight: 400;
    color: #c8b880;
    letter-spacing: -0.01em;
}
.topbar-brand .ring {
    width: 22px; height: 22px;
    border-radius: 50%;
    border: 2px solid #c49040;
    box-shadow: 0 0 6px rgba(196,144,64,0.4);
    flex-shrink: 0;
}

/* ── Settings panel styling (Streamlit widgets inside) ── */
.cfg-panel {
    background: #141210;
    border-bottom: 1px solid #1c1a16;
    padding: 0.7rem 0;
}
.cfg-panel label { color: #6a6055 !important; font-size: 0.75rem !important; }
.cfg-panel [data-testid="stSelectbox"] > div > div,
.cfg-panel [data-testid="stTextInput"] > div > div > input {
    background: #1c1a16 !important;
    border: 1px solid #2a2620 !important;
    border-radius: 8px !important;
    color: #c8c0b4 !important;
    font-size: 0.84rem !important;
    padding: 0.3rem 0.7rem !important;
}
.cfg-panel button[kind="secondary"],
.cfg-panel button[kind="primary"] {
    background: #2a2418 !important;
    border: 1px solid #3e3222 !important;
    border-radius: 8px !important;
    color: #c0903a !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    padding: 0.35rem 0.9rem !important;
    transition: background 0.18s !important;
    white-space: nowrap !important;
}
.cfg-panel button:hover { background: #3e3222 !important; }

/* ── Status chip ── */
.status-chip {
    font-size: 0.68rem;
    color: #7a9060;
    background: #121810;
    border: 1px solid #253020;
    border-radius: 20px;
    padding: 0.12em 0.6em;
    display: inline-block;
    margin: 0.3rem 0 0.6rem;
}
.status-chip-warn { background: #181408; border-color: #352808; color: #907840; }

/* ── Empty / welcome state ── */
.welcome-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 8vh 0 4vh;
    gap: 1.2rem;
}
.welcome-title {
    font-size: 1.6rem;
    font-weight: 300;
    color: #c8b880;
    text-align: center;
    line-height: 1.3;
}
.welcome-sub {
    font-size: 0.84rem;
    color: #5a5448;
    text-align: center;
    max-width: 420px;
}
.suggestion-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    width: 100%;
    max-width: 520px;
    margin-top: 0.5rem;
}
.suggestion-card {
    background: #161412;
    border: 1px solid #252118;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    font-size: 0.82rem;
    color: #a09880;
    line-height: 1.5;
}
.suggestion-card:hover { border-color: #4a3e28; background: #1c1a14; color: #c8b880; }
.suggestion-card .card-title {
    font-weight: 500;
    color: #c0b090;
    margin-bottom: 0.2rem;
    font-size: 0.84rem;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
    align-items: flex-start !important;
    max-width: 100% !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    font-size: 0.93rem !important;
    line-height: 1.85 !important;
    color: #c4bbb0 !important;
    font-weight: 300 !important;
}
[data-testid="stChatMessage"] strong { color: #ddd4c4 !important; font-weight: 500 !important; }
[data-testid="stChatMessage"] code {
    background: #1e1c18 !important;
    color: #b09050 !important;
    border-radius: 4px !important;
    padding: 0.1em 0.4em !important;
    font-size: 0.83em !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
[data-testid="stMarkdownContainer"] {
    background: #1a1814 !important;
    border: 1px solid #252118 !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 0.75rem 1.1rem !important;
    max-width: 82%;
    margin-left: auto;
}
/* Assistant -- no bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
[data-testid="stMarkdownContainer"] {
    padding: 0.2rem 0 !important;
    background: transparent !important;
}
[data-testid="chatAvatarIcon-user"] img,
[data-testid="chatAvatarIcon-assistant"] img {
    width: 28px !important;
    height: 28px !important;
    border-radius: 50% !important;
}

/* ── Input bar ── */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div {
    background: #0f0e0c !important;
    background-image: none !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stBottom"] { padding: 0 !important; }
[data-testid="stBottom"] > div {
    padding: 0.8rem 0 1.5rem !important;
    max-width: 780px;
    margin: 0 auto;
}
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div {
    background: #1a1814 !important;
}
[data-testid="stChatInput"] {
    border: 1px solid #2a2620 !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.4) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #5a4828 !important;
    box-shadow: 0 2px 20px rgba(0,0,0,0.5) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #c8c0b4 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
    font-weight: 300 !important;
    line-height: 1.6 !important;
    caret-color: #c0903a !important;
    border: none !important;
    outline: none !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #3e3830 !important; opacity: 1 !important; }
[data-testid="stChatInput"] button {
    background: #252018 !important;
    border-radius: 9px !important;
    border: none !important;
    margin: 4px !important;
    transition: background 0.18s !important;
}
[data-testid="stChatInput"] button:hover { background: #3a3020 !important; }
[data-testid="stChatInput"] button svg { fill: #c0903a !important; color: #c0903a !important; }

/* ── Generated images ── */
[data-testid="stImage"] img {
    border-radius: 10px !important;
    border: 1px solid #22201a !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.6) !important;
    margin-top: 0.5rem;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: #180e0e !important;
    border: 1px solid #3a1c1c !important;
    border-radius: 10px !important;
    color: #a06060 !important;
}

/* ── Topbar Streamlit column alignment ── */
[data-testid="stHorizontalBlock"] { align-items: center !important; gap: 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

#Helpers

def _provider_from_model(model_id: str) -> str:
    if model_id.startswith("google") or model_id.startswith("gemini"):
        return "gemini"
    if model_id.startswith("openai"):
        return "openai"
    return "ollama"


def _init_agent(model_id: str, api_key_override: str | None = None) -> None:
    provider = _provider_from_model(model_id)
    if provider in ("gemini", "openai"):
        rotator = KeyRotator(provider)
        rotator.inject(override=api_key_override or None)
    else:
        rotator = None

    deps  = AgentDependencies()
    agent = create_lensing_agent(model_name=model_id, deps=deps)

    st.session_state.agent            = agent
    st.session_state.deps             = deps
    st.session_state.message_history  = []
    st.session_state.chat_log         = []
    st.session_state.rotator          = rotator
    st.session_state.current_model    = model_id
    st.session_state.api_key_override = api_key_override

    if "async_loop" not in st.session_state:
        st.session_state.async_loop = asyncio.new_event_loop()


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("quota", "429", "exhausted", "rate limit", "resource_exhausted"))


def _extract_tool_interactions(new_messages) -> list[dict]:
    """Extract tool call + result pairs from pydantic-ai new_messages."""
    import json
    tool_results: dict[str, str] = {}
    interactions: list[dict] = []

    # First pass: collect tool returns (ToolReturnPart)
    for msg in new_messages:
        for part in getattr(msg, "parts", []):
            call_id = getattr(part, "tool_call_id", None)
            content  = getattr(part, "content", None)
            # ToolReturnPart has tool_call_id + content but NOT args
            if call_id and content is not None and not hasattr(part, "args"):
                tool_results[call_id] = str(content)

    # Second pass: collect tool calls (ToolCallPart) and pair with results
    for msg in new_messages:
        for part in getattr(msg, "parts", []):
            tool_name = getattr(part, "tool_name", None)
            args      = getattr(part, "args", None)
            call_id   = getattr(part, "tool_call_id", None)
            # ToolCallPart has tool_name + args (not content)
            if tool_name and args is not None:
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                result = tool_results.get(call_id or "", "")
                interactions.append({
                    "name":   tool_name,
                    "args":   args,
                    "result": result,
                })
    return interactions


model_labels = [label for _, label in SUPPORTED_MODELS]
model_ids    = [mid   for mid, _  in SUPPORTED_MODELS]

#Session init 
if "agent" not in st.session_state:
    env_model     = os.environ.get("LENSING_AGENT_MODEL")
    startup_model = env_model if env_model else model_ids[0]
    _init_agent(startup_model)
    try:
        st.session_state.settings_model_idx = model_ids.index(startup_model)
    except ValueError:
        st.session_state.settings_model_idx = 0
    st.session_state.show_settings = False

#Top nav bar 
nav_l, nav_r = st.columns([4, 1])

with nav_l:
    st.markdown("""
    <div class="topbar-brand">
      <div class="ring"></div>
      DeepLense Simulator
    </div>
    """, unsafe_allow_html=True)

with nav_r:
    col_gear, col_clear = st.columns([1, 1], gap="small")
    with col_gear:
        if st.button("⚙", key="btn_settings", help="Model settings", use_container_width=True):
            st.session_state.show_settings = not st.session_state.get("show_settings", False)
    with col_clear:
        if st.button("↺", key="btn_clear", help="Clear conversation", use_container_width=True):
            st.session_state.message_history = []
            st.session_state.chat_log = []
            st.rerun()

st.markdown('<div style="border-bottom:1px solid #1c1a16;margin-bottom:0"></div>', unsafe_allow_html=True)

#Settings panel (toggleable)
if st.session_state.get("show_settings", False):
    with st.container():
        st.markdown('<div class="cfg-panel">', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns([3, 2, 3, 1], gap="small")

        with s1:
            selected_label = st.selectbox(
                "Model",
                options=model_labels,
                index=st.session_state.get("settings_model_idx", 0),
                key="cfg_model",
                label_visibility="collapsed",
            )
            selected_model_id = model_ids[model_labels.index(selected_label)]

        with s2:
            if selected_model_id == "ollama:custom":
                ollama_name = st.text_input(
                    "Ollama model",
                    value=st.session_state.get("ollama_model_name", "llama3.2"),
                    placeholder="llama3.2",
                    key="cfg_ollama",
                    label_visibility="collapsed",
                )
            else:
                ollama_name = ""
                st.write("")

        with s3:
            api_key_input = st.text_input(
                "API key",
                type="password",
                value="",
                placeholder="API key override (optional)",
                key="cfg_apikey",
                label_visibility="collapsed",
            )
            api_key_override = api_key_input.strip() or None

        with s4:
            apply = st.button("Apply", key="cfg_apply", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Resolve model id
        if selected_model_id == "ollama:custom":
            resolved_id = f"ollama:{ollama_name.strip()}" if ollama_name.strip() else "ollama:llama3.2"
        else:
            resolved_id = selected_model_id

        # Status chip
        rotator = st.session_state.get("rotator")
        if api_key_override:
            key_lbl   = "key from input"
            chip_cls  = "status-chip"
        elif rotator:
            key_lbl   = rotator.status_label()
            chip_cls  = "status-chip" if rotator.current_index == 0 else "status-chip-warn"
        else:
            key_lbl   = "Ollama (no key)"
            chip_cls  = "status-chip"

        active = st.session_state.get("current_model", resolved_id)
        st.markdown(f'<span class="{chip_cls}">{active} &nbsp;·&nbsp; {key_lbl}</span>',
                    unsafe_allow_html=True)

        if apply:
            prev = st.session_state.get("current_model", "")
            prev_key = st.session_state.get("api_key_override")
            if resolved_id != prev or api_key_override != prev_key:
                _init_agent(resolved_id, api_key_override=api_key_override)
                st.session_state["ollama_model_name"]  = ollama_name
                st.session_state["settings_model_idx"] = model_ids.index(selected_model_id)
                st.toast(f"Agent ready: {resolved_id}")
            st.session_state.show_settings = False
            st.rerun()

# Suggestion chips (empty state) 
SUGGESTIONS = [
    ("Galaxy-scale CDM lens",   "Generate 5 CDM lensing images using Model_I with default parameters"),
    ("Axion fuzzy dark matter", "Suggest parameters for a low mass axion simulation"),
    ("Euclid high-z source",    "Generate 3 axion lensing images with Model_II at high redshift"),
    ("Statistical ensemble",    "Suggest parameters for a statistical study"),
]

if not st.session_state.chat_log:
    st.markdown("""
    <div class="welcome-wrap">
      <div class="welcome-title">What would you like to simulate?</div>
      <div class="welcome-sub">
        Describe a gravitational lensing configuration in plain English.<br>
        The agent will ask follow-up questions, then generate images.
      </div>
      <div class="suggestion-grid">
    """ + "".join(
        f'<div class="suggestion-card"><div class="card-title">{title}</div>{desc}</div>'
        for title, desc in SUGGESTIONS
    ) + """
      </div>
    </div>
    """, unsafe_allow_html=True)

#Chat history
for msg in st.session_state.chat_log:
    av = ASSISTANT_AVATAR if msg["role"] == "assistant" else USER_AVATAR
    with st.chat_message(msg["role"], avatar=av):
        st.markdown(msg["content"])
        if msg.get("images"):
            imgs = msg["images"]
            if len(imgs) == 1:
                st.image(str(imgs[0]), width=320)
            else:
                cols = st.columns(min(len(imgs), 3))
                for i, p in enumerate(imgs):
                    cols[i % 3].image(str(p), width="stretch")

# Chat input and run
if user_input := st.chat_input("Describe a lensing simulation..."):
    st.session_state.chat_log.append({"role": "user", "content": user_input, "images": []})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner(""):
            out_dir = Path(st.session_state.deps.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            before = {p: p.stat().st_mtime for p in out_dir.glob("*.png")}

            loop = st.session_state.async_loop
            asyncio.set_event_loop(loop)

            result = None
            error  = None

            for attempt in range(4):
                try:
                    result = loop.run_until_complete(
                        st.session_state.agent.run(
                            user_input,
                            deps=st.session_state.deps,
                            message_history=st.session_state.message_history,
                        )
                    )
                    break
                except Exception as exc:
                    if _is_quota_error(exc):
                        rot = st.session_state.get("rotator")
                        if rot and rot.rotate():
                            st.toast(f"Quota hit. Rotating to key {rot.current_index+1}/{rot.total_keys}...", icon="🔄")
                            _init_agent(
                                st.session_state.current_model,
                                api_key_override=st.session_state.get("api_key_override"),
                            )
                        else:
                            error = "All API keys exhausted. Add more keys to .env or enter one via ⚙."
                            break
                    else:
                        error = str(exc)
                        break

            if result:
                st.session_state.message_history = result.all_messages()
                content = result.output

                after   = {p: p.stat().st_mtime for p in out_dir.glob("*.png")}
                new_img = sorted([p for p, t in after.items() if p not in before or t > before[p]])
                paths   = [str(p) for p in new_img]

                # ── Thinking dropdown ──
                try:
                    interactions = _extract_tool_interactions(result.new_messages())
                    if interactions:
                        with st.expander(f"🔍 Agent reasoning ({len(interactions)} step{'s' if len(interactions) != 1 else ''})", expanded=False):
                            for step in interactions:
                                tool_icon = {
                                    "validate_parameters":  "✅",
                                    "suggest_parameters":   "💡",
                                    "get_model_info":       "📐",
                                    "generate_lensing_images": "🔭",
                                }.get(step["name"], "🔧")
                                st.markdown(f"{tool_icon} **`{step['name']}`**")
                                if step["args"] and step["args"] != {}:
                                    args_clean = {k: v for k, v in step["args"].items() if v is not None}
                                    if args_clean:
                                        st.json(args_clean, expanded=False)
                                if step["result"]:
                                    st.markdown(
                                        f"<div style='font-size:0.8rem;color:#7a7268;background:#141210;border:1px solid #1e1c18;"
                                        f"border-radius:8px;padding:0.5rem 0.75rem;margin:0.3rem 0 0.6rem;white-space:pre-wrap;'>"
                                        f"{step['result'][:600]}{'...' if len(step['result'])>600 else ''}</div>",
                                        unsafe_allow_html=True,
                                    )
                except Exception:
                    pass 

            
                st.markdown(content)
                if len(paths) == 1:
                    st.image(paths[0], width=320)
                elif paths:
                    cols = st.columns(min(len(paths), 3))
                    for i, p in enumerate(paths):
                        cols[i % 3].image(p, width="stretch")

                st.session_state.chat_log.append({
                    "role":    "assistant",
                    "content": content,
                    "images":  paths,
                })
            else:
                st.error(f"Error: {error}")