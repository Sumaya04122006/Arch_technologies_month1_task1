"""
Local AI Studio
A private, offline ChatGPT-style workspace powered by Ollama + Streamlit.

Features: live model detection, AI personas, streaming replies, tokens/sec
stats, multiple saved chats (persisted to disk), export, and custom UI.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

# ------------------------------------------------------------------ config
OLLAMA = "http://localhost:11434"
DATA_FILE = Path("chats.json")

PERSONAS = {
    "🧠 General Assistant": {
        "prompt": "You are a helpful, concise and accurate assistant.",
        "starters": [
            "Explain how neural networks learn, simply",
            "Give me a 3-day study plan for learning Python",
            "What are the pros and cons of running LLMs locally?",
            "Write a short email asking for an internship extension",
        ],
    },
    "🩺 Medical Explainer": {
        "prompt": (
            "You are a medical education assistant. Explain clearly using plain "
            "language, then give key terms. Always end with one line reminding the "
            "user this is educational information, not medical advice."
        ),
        "starters": [
            "What are the symptoms of type 2 diabetes?",
            "How does metformin work?",
            "Explain hypertension like I'm a first-year student",
            "Difference between bacterial and viral infections?",
        ],
    },
    "👨‍💻 Code Mentor": {
        "prompt": (
            "You are a senior software engineer mentoring a junior. Give working "
            "code in markdown code blocks, then explain it step by step."
        ),
        "starters": [
            "Write a Python function to detect palindromes",
            "Explain list comprehensions with examples",
            "How do I read a CSV file with pandas?",
            "What is the difference between a list and a tuple?",
        ],
    },
    "🧒 ELI5 Teacher": {
        "prompt": "Explain everything as if to a curious 5-year-old, using fun analogies.",
        "starters": [
            "How does the internet work?",
            "What is artificial intelligence?",
            "Why is the sky blue?",
            "What is a black hole?",
        ],
    },
    "🏴‍☠️ Pirate Captain": {
        "prompt": "You are a witty pirate captain. Answer helpfully but always in pirate speak.",
        "starters": [
            "Explain machine learning",
            "How do I stay productive?",
            "Tips for a job interview?",
            "What is cloud computing?",
        ],
    },
}

st.set_page_config(page_title="Local AI Studio", page_icon="⚡", layout="wide")

# --------------------------------------------------------------------- css
st.markdown(
    """
<style>
.hero {
    padding: 1.2rem 1.5rem; border-radius: 16px; margin-bottom: 1rem;
    background: linear-gradient(120deg, #6a11cb 0%, #2575fc 55%, #00c6ff 100%);
    color: white;
}
.hero h1 { margin: 0; font-size: 2rem; color: white; }
.hero p { margin: .2rem 0 0 0; opacity: .9; }
.pill {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: .75rem; background: rgba(255,255,255,.22); margin-right: 6px;
}
.stat { color: #8b93a7; font-size: .78rem; margin-top: -6px; }
div[data-testid="stChatMessage"] { border-radius: 14px; }
section[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.2); }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- storage
def load_chats():
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_chats():
    DATA_FILE.write_text(
        json.dumps(st.session_state.chats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def new_chat():
    cid = uuid.uuid4().hex[:8]
    st.session_state.chats[cid] = {
        "title": "New chat",
        "created": datetime.now().isoformat(),
        "messages": [],
    }
    st.session_state.current = cid
    save_chats()


if "chats" not in st.session_state:
    st.session_state.chats = load_chats()
if not st.session_state.chats:
    new_chat()
if "current" not in st.session_state or st.session_state.current not in st.session_state.chats:
    st.session_state.current = max(
        st.session_state.chats, key=lambda c: st.session_state.chats[c]["created"]
    )


# ------------------------------------------------------------------ ollama
def get_models():
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=2)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


def stream_chat(model, messages, temperature, max_tokens):
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    with requests.post(f"{OLLAMA}/api/chat", json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                yield json.loads(line)


# ----------------------------------------------------------------- sidebar
models = get_models()

with st.sidebar:
    st.markdown("## ⚡ Local AI Studio")
    if models is None:
        st.error("🔴 Ollama offline. Open the Ollama app.")
    elif not models:
        st.warning("🟡 Ollama is running but no models installed.")
    else:
        st.success(f"🟢 Ollama online · {len(models)} model(s)")

    if st.button("➕ New chat", use_container_width=True, type="primary"):
        new_chat()
        st.rerun()

    st.markdown("### 💬 Chats")
    ordered = sorted(
        st.session_state.chats.items(), key=lambda kv: kv[1]["created"], reverse=True
    )
    for cid, chat in ordered:
        label = ("● " if cid == st.session_state.current else "") + chat["title"][:28]
        if st.button(label, key=f"chat_{cid}", use_container_width=True):
            st.session_state.current = cid
            st.rerun()

    st.divider()
    st.markdown("### ⚙️ Settings")
    model = st.selectbox("Model", models or ["(no model)"])
    persona_name = st.selectbox("Persona", list(PERSONAS.keys()))
    temperature = st.slider("Creativity", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max response length", 64, 2048, 512, 64)

    st.divider()
    chat = st.session_state.chats[st.session_state.current]

    md = f"# {chat['title']}\n\n" + "\n\n".join(
        f"**{'You' if m['role'] == 'user' else 'AI'}:** {m['content']}"
        for m in chat["messages"]
    )
    st.download_button(
        "⬇️ Export chat (.md)", md, file_name="chat.md", use_container_width=True
    )
    c1, c2 = st.columns(2)
    if c1.button("🔄 Clear", use_container_width=True):
        chat["messages"] = []
        chat["title"] = "New chat"
        save_chats()
        st.rerun()
    if c2.button("🗑️ Delete", use_container_width=True):
        del st.session_state.chats[st.session_state.current]
        if not st.session_state.chats:
            new_chat()
        else:
            st.session_state.current = next(iter(st.session_state.chats))
        save_chats()
        st.rerun()

# -------------------------------------------------------------------- main
persona = PERSONAS[persona_name]
chat = st.session_state.chats[st.session_state.current]

st.markdown(
    f"""
<div class="hero">
  <h1>⚡ Local AI Studio</h1>
  <p>Private. Offline. Yours. Every word is generated on your own machine.</p>
  <p style="margin-top:.6rem">
    <span class="pill">{model}</span>
    <span class="pill">{persona_name}</span>
    <span class="pill">temp {temperature}</span>
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# Render history
for m in chat["messages"]:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
        st.markdown(m["content"])
        if m.get("stats"):
            st.markdown(f"<div class='stat'>{m['stats']}</div>", unsafe_allow_html=True)

# Starter prompts on empty chat
pending = None
if not chat["messages"]:
    st.markdown("#### Try one of these")
    cols = st.columns(2)
    for i, s in enumerate(persona["starters"]):
        if cols[i % 2].button(s, key=f"starter_{i}", use_container_width=True):
            pending = s

prompt = st.chat_input("Message your local AI...") or pending

if prompt:
    if not models:
        st.error("No Ollama model available. Start Ollama and run: ollama pull llama3.2:3b")
        st.stop()

    chat["messages"].append({"role": "user", "content": prompt})
    if chat["title"] == "New chat":
        chat["title"] = prompt[:40]

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Build payload: system prompt + history (strip extra keys like "stats")
    payload_msgs = [{"role": "system", "content": persona["prompt"]}] + [
        {"role": m["role"], "content": m["content"]} for m in chat["messages"]
    ]

    with st.chat_message("assistant", avatar="🤖"):
        box = st.empty()
        text, stats = "", ""
        start = time.time()
        try:
            for chunk in stream_chat(model, payload_msgs, temperature, max_tokens):
                text += chunk.get("message", {}).get("content", "")
                box.markdown(text + " ▌")
                if chunk.get("done"):
                    toks = chunk.get("eval_count", 0)
                    dur = chunk.get("eval_duration", 0) / 1e9
                    tps = toks / dur if dur else 0
                    stats = (
                        f"⚡ {tps:.1f} tok/s · {toks} tokens · "
                        f"{time.time() - start:.1f}s · {model}"
                    )
            box.markdown(text)
            if stats:
                st.markdown(f"<div class='stat'>{stats}</div>", unsafe_allow_html=True)
        except requests.exceptions.ConnectionError:
            text = "⚠️ Cannot reach Ollama. Make sure the Ollama app is running."
            box.error(text)
        except requests.exceptions.HTTPError as e:
            text = f"⚠️ Ollama error: {e}. Is the model '{model}' installed?"
            box.error(text)

    chat["messages"].append({"role": "assistant", "content": text, "stats": stats})
    save_chats()
    st.rerun()