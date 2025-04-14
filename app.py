import streamlit as st
from PIL import Image
from datetime import datetime
import time
import json
from gemini_test2 import (
    classifier,
    critic,
    ask_gemin_ImageAnalyzer,
    tenancy_faq,
    answer_verification,
    ChatState
)

st.set_page_config(page_title="🏡 Real Estate Chatbot", layout="centered")
st.title("🏡 Real Estate Chatbot")
st.caption("Ask about property issues or tenancy questions. I'm your smart assistant 🤖")

# Reset
if st.button("🔁 Restart"):
    st.session_state.chat_state = {
        "image": None, "message": "", "location": None, "stage": None,
        "category": None, "response": None, "conversation_history": [],
        "intermediate_responses": [], "requires_human_input": False,
        "clarification_question": None, "clarification_attempts": 0,
        "exit_requested": False
    }
    st.session_state.messages = []
    st.rerun()

# Initialize
if "chat_state" not in st.session_state:
    st.session_state.chat_state = {
        "image": None, "message": "", "location": None, "stage": None,
        "category": None, "response": None, "conversation_history": [],
        "intermediate_responses": [], "requires_human_input": False,
        "clarification_question": None, "clarification_attempts": 0,
        "exit_requested": False
    }
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Upload image
uploaded_image = st.file_uploader("📷 Upload property image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    st.image(uploaded_image, caption="Preview", use_container_width=True)
    uploaded_image.seek(0)
    st.session_state.chat_state["image"] = uploaded_image.read()

# Check mode
clarify_mode = st.session_state.chat_state.get("requires_human_input", False)
placeholder = (
    f"✏️ {st.session_state.chat_state.get('clarification_question')} (or type 'exit')" if clarify_mode
    else "💬 Ask your property or tenancy-related question..."
)

# Clarification input below last assistant response
if clarify_mode:
    st.markdown("### 🤔 The previous response might need more detail.")
    st.markdown("**How can I further assist you?**")

# Input
user_input = st.text_input("Your input", placeholder="", key="user_input_box") if clarify_mode else st.chat_input(placeholder)

# Process input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    chat = st.session_state.chat_state

    if clarify_mode:
        if user_input.strip().lower() == "exit":
            chat["requires_human_input"] = False
            chat["clarification_question"] = None
            chat["exit_requested"] = True
        else:
            chat["message"] += " " + user_input
            chat["clarification_question"] = None
            chat["requires_human_input"] = False
            chat["clarification_attempts"] += 1
    else:
        chat["message"] = user_input.strip()
        chat["clarification_attempts"] = 0
        chat["exit_requested"] = False
        chat["intermediate_responses"] = []

    with st.chat_message("assistant"):
        # Classifier
        st.markdown("#### 🧠 Classifier Agent")
        start = time.time()
        chat = classifier(chat)
        elapsed = time.time() - start
        st.markdown(f"🟢 **Status:** Done · ⏱️ {elapsed:.2f}s")
        st.code(json.dumps(chat["intermediate_responses"][-1], indent=2))

        # Critic
        st.markdown("#### 🔍 Critic Agent")
        start = time.time()
        chat = critic(chat)
        elapsed = time.time() - start
        st.markdown(f"🟢 **Status:** Done · ⏱️ {elapsed:.2f}s")
        st.code(json.dumps(chat["intermediate_responses"][-1], indent=2))

        # Clarify again?
        if chat.get("requires_human_input"):
            q = chat.get("clarification_question", "Can you explain more?")
            st.session_state.messages.append({"role": "assistant", "content": f"🤖 {q}"})
            st.markdown(f"🤖 _{q}_")
            st.rerun()

        # Agent Routing
        st.markdown("#### 🔧 Execution Agent")
        if chat.get("category") == "issue_detection":
            if not chat.get("image"):
                chat["response"] = "⚠️ I need an image to analyze this issue. Please upload it."
            else:
                chat = ask_gemin_ImageAnalyzer(chat)
        elif chat.get("category") == "faq":
            chat = tenancy_faq(chat)
        else:
            chat["response"] = "❌ Sorry, I couldn't understand your query."

        # Verifier
        st.markdown("#### ✅ Verifier Agent")
        chat = answer_verification(chat)

        if chat.get("requires_human_input"):
            q = chat.get("clarification_question", "Need more detail.")
            st.session_state.messages.append({"role": "assistant", "content": f"🤖 {q}"})
            st.markdown(f"🤖 _{q}_")
            st.rerun()

        # Final Response
        st.markdown("### 🎯 Final Response")
        final_response = chat.get("response", "✅ Done!")
        st.success(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
