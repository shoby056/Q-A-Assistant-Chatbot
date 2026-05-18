import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="Simple LangChain Chatbot With Groq",
    page_icon="🚀"
)

st.title("🚀 Simple LangChain Chatbot With Groq")
st.markdown("Learn LangChain basics with Groq")

# Sidebar
with st.sidebar:
    st.header("Settings")

    # API KEY INPUT (local use)
    api_key_input = st.text_input(
        "Groq API Key (optional if using Streamlit Secrets)",
        type="password",
        help="Get free API key at console.groq.com"
    )

    model_name = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ],
        index=0
    )

# Use API key (Secrets first, then input)
api_key = os.getenv("GROQ_API_KEY") or api_key_input

# Clear chat
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# LLM chain
@st.cache_resource
def get_chain(api_key, model_name):

    if not api_key:
        return None

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        streaming=True
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer clearly and concisely."),
        ("user", "{question}")
    ])

    return prompt | llm | StrOutputParser()

chain = get_chain(api_key, model_name)

# API missing warning
if not api_key:
    st.warning("⚠️ Please enter API key or add it in Streamlit Secrets")
    st.stop()

# Chat history display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if question := st.chat_input("Ask me anything"):

    # user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    # assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            for chunk in chain.stream({"question": question}):
                full_response += chunk
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

        except Exception as e:
            st.error(str(e))

# Examples section (RESTORED)
st.markdown("---")
st.markdown("### 💡 Try these Examples:")

col1, col2 = st.columns(2)

with col1:
    st.markdown("- What is LangChain?")
    st.markdown("- Explain Groq technology")

with col2:
    st.markdown("- How do I learn programming?")
    st.markdown("- Write a haiku about AI")

# Footer
st.markdown("---")
st.markdown("Built with LangChain + Groq 🚀")
