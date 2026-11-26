import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from graph import graph

st.set_page_config(page_title="India Union Budget 2024-25 Chatbot", layout="centered")
st.title("India Union Budget 2024-25 Chatbot")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Session")
    if st.button("New chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Thread ID: {st.session_state.thread_id[:8]}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.write(f"- {src}")

if prompt := st.chat_input("Ask about the India Union Budget 2024-25..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.chat_message("assistant"):
        with st.spinner("Searching budget documents..."):
            result = graph.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
            )
            answer = result["messages"][-1].content
            sources = result.get("sources", [])
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.write(f"- {src}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )