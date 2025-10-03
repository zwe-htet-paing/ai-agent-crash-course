import streamlit as st
import asyncio
from typing import Any, Dict, List

import ingest
import search_agent
import logs

# --- Utility: Safe asyncio runner ---
def run_async(coro):
    """Run async functions safely in Streamlit (avoids nested event loops)."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

# --- Initialization ---
@st.cache_resource(show_spinner=True)
def initialize_agent():
    repo_owner = "pydantic"
    repo_name = "pydantic-ai"
    chunk_method = "markdown_sections"

    try:
        index = ingest.index_data(repo_owner, repo_name, chunk_method=chunk_method)
        agent = search_agent.init_agent(index, repo_owner, repo_name)
        return agent

    except Exception as e:
        st.error("❌ Initialization Error: Could not load the knowledge base.")
        st.exception(e)
        st.stop()

# --- Chat Rendering ---
def render_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg['content'])

# --- UI Styling ---
st.set_page_config(page_title="Pydantic-ai Assistant", page_icon="🤖", layout="centered")
st.markdown(
    """
    <style>
        .main {background-color: #f9fafb;}
        .stChatMessage {border-radius: 12px; padding: 8px; margin-bottom: 10px;}
        .stButton button {border-radius: 10px; background-color: #1E90FF; color: white; font-weight: bold; margin-bottom: 5px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.title("🤖 Pydantic-ai Assistant")
st.caption("Ask me anything about the **pydantic/pydantic-ai** repository.")

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

# --- Initialize system ---
agent = initialize_agent()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Render chat history ---
render_chat_history()

# # --- Chat input ---
# example_question = "How do I install pydantic-ai?"
# if prompt := st.chat_input(f"💬 Type your question here... e.g. '{example_question}'"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     try:
#         with st.chat_message("assistant"):
#             with st.spinner("🧠 Thinking..."):
#                 response = run_async(agent.run(user_prompt=prompt))
#                 answer = response.output
#                 st.markdown(answer)

#         st.session_state.messages.append({"role": "assistant", "content": answer})
#         logs.log_interaction_to_file(agent, response.new_messages())

#     except Exception as e:
#         error_message = (
#             "⚠️ I encountered an error while processing your request. "
#             "Please try again or check the system logs."
#         )
#         st.error(error_message)
#         st.session_state.messages.append({"role": "assistant", "content": error_message})
#         print(f"Agent Execution Error: {e}")

# --- Render chat history ---
render_chat_history()

# --- Streaming helper ---
def stream_response(prompt: str):
    async def agen():
        async with agent.run_stream(user_prompt=prompt) as result:
            last_len = 0
            full_text = ""
            async for chunk in result.stream_output(debounce_by=0.01):
                # stream only the delta
                new_text = chunk[last_len:]
                last_len = len(chunk)
                full_text = chunk
                if new_text:
                    yield new_text
            # log once complete
            logs.log_interaction_to_file(agent, result.new_messages())
            st.session_state._last_response = full_text
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    agen_obj = agen()
    try:
        while True:
            piece = loop.run_until_complete(agen_obj.__anext__())
            yield piece
    except StopAsyncIteration:
        return

# --- Chat input ---
if prompt := st.chat_input("💬 Ask your question..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Assistant message (streamed)
    with st.chat_message("assistant"):
        response_text = st.write_stream(stream_response(prompt))
    
    # Save full response to history
    final_text = getattr(st.session_state, "_last_response", response_text)
    st.session_state.messages.append({"role": "assistant", "content": final_text})