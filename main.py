import streamlit as st

# Updated imports for HuggingFace
from services.get_models_list import get_hf_models_list
from services.get_title import get_chat_title
from services.chat_utilities import get_answer
from db.conversations import (
    create_new_conversation,
    add_message,
    get_conversation,
    get_all_conversations,
)

# ---- Page setup ----
st.set_page_config(page_title="ChatBot", page_icon="💬", layout="centered")
st.title("🤖 ChatBot")

# ---- Models ----
if "HF_MODELS" not in st.session_state:
    st.session_state.HF_MODELS = get_hf_models_list()

selected_model = st.selectbox("Select Model", st.session_state.HF_MODELS)

# ---- Session state ----
st.session_state.setdefault("conversation_id", None)
st.session_state.setdefault("conversation_title", None)
st.session_state.setdefault("chat_history", [])  # [{role, content}]

# ---- Sidebar: conversations ----
with st.sidebar:
    st.header("💬 Chat History")

    conversations = get_all_conversations()  # {conv_id: title}

    if st.button("➕ New Chat"):
        st.session_state.conversation_id = None
        st.session_state.conversation_title = None
        st.session_state.chat_history = []

    for cid, title in conversations.items():
        is_current = cid == st.session_state.conversation_id
        label = f" {title}" if is_current else title
        if st.button(label, key=f"conv_{cid}"):
            doc = get_conversation(cid) or {}
            st.session_state.conversation_id = cid
            st.session_state.conversation_title = doc.get("title", "Untitled")
            st.session_state.chat_history = [
                {"role": m["role"], "content": m["content"]} for m in doc.get("messages", [])
            ]

# ---- Show chat so far ----
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Chat input ----
user_query = st.chat_input("Ask AI...")
if user_query:
    # 1) Show + store user message in UI state
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # 2) Persist to DB (create convo on first message, else append)
    if st.session_state.conversation_id is None:
        try:
            title = get_chat_title(selected_model, user_query) or "New Chat"
        except Exception:
            title = "New Chat"
        conv_id = create_new_conversation(title=title, role="user", content=user_query)
        st.session_state.conversation_id = conv_id
        st.session_state.conversation_title = title
    else:
        add_message(st.session_state.conversation_id, "user", user_query)

    # 3) Get assistant response using HuggingFace LLM
    try:
        assistant_text = get_answer(selected_model, st.session_state.chat_history)
        # Debug print to ensure text is captured
        print("Assistant text:", repr(assistant_text))
    except Exception as e:
        assistant_text = f"[Error getting response: {e}]"

    # 4) Show + store assistant message
    with st.chat_message("assistant"):
        st.markdown(assistant_text if assistant_text else "[No response]")  # fallback display
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})

    # 5) Persist assistant message
    if st.session_state.conversation_id:
        add_message(st.session_state.conversation_id, "assistant", assistant_text)