# src/app.py
from __future__ import annotations

import os
import streamlit as st

from agent import AgentConfig, StudyCopilotAgent


st.set_page_config(page_title="StudyCopilot (Agent + RAG)", layout="wide")

st.title("📚 StudyCopilot — Minimal Agent + RAG (Ollama)")
st.caption("Local Ollama + Chroma vectordb + strict evidence gate")

with st.sidebar:
    st.header("Settings")

    ollama_llm_model = st.text_input("Ollama LLM model", value=os.getenv("OLLAMA_LLM_MODEL", "llama3"))
    ollama_embed_model = st.text_input(
        "Ollama embedding model",
        value=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        help="If you don't have nomic-embed-text, install/pull it in Ollama or set to an available embedding model.",
    )
    vectordb_dir = st.text_input("vectordb directory", value=os.getenv("VECTORDB_DIR", "vectordb"))

    top_k = st.slider("top_k (retrieve chunks)", min_value=1, max_value=10, value=int(os.getenv("TOP_K", "4")))
    temperature = st.slider("temperature", min_value=0.0, max_value=1.0, value=float(os.getenv("TEMP", "0.2")))

    st.divider()
    st.write("**Run prerequisites**")
    st.code("ollama serve\nollama pull llama3\nollama pull nomic-embed-text", language="bash")

# lazy init agent in session
if "agent" not in st.session_state:
    cfg = AgentConfig(
        ollama_llm_model=ollama_llm_model,
        vectordb_dir=vectordb_dir,
        ollama_embedding_model=ollama_embed_model,
        top_k=top_k,
        temperature=temperature,
    )
    st.session_state.agent = StudyCopilotAgent(cfg)

# if user changes settings, recreate agent
if st.sidebar.button("Rebuild Agent"):
    cfg = AgentConfig(
        ollama_llm_model=ollama_llm_model,
        vectordb_dir=vectordb_dir,
        ollama_embedding_model=ollama_embed_model,
        top_k=top_k,
        temperature=temperature,
    )
    st.session_state.agent = StudyCopilotAgent(cfg)
    st.success("Agent rebuilt.")


if "messages" not in st.session_state:
    st.session_state.messages = []

# chat display
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a question about your notes...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking (Agent → Retrieve → Evidence Gate → Answer)..."):
            result = st.session_state.agent.answer(prompt)

        st.markdown(result["final"])

        with st.expander("Debug (route / citations / docs_found)"):
            st.write(f"**route:** {result['route']}")
            st.write(f"**docs_found:** {result['docs_found']}")
            if result["citations"]:
                st.write("**citations:**")
                for c in result["citations"]:
                    st.write(c)

    st.session_state.messages.append({"role": "assistant", "content": result["final"]})