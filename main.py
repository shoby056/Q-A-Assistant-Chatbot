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

# Title
st.title("🚀 Simple LangChain Chatbot With Groq")
st.markdown("Learn LangChain basics with Groq's ultra-fast inference")

# Sidebar
with st.sidebar:
    st.header("Settings")

    # API key input
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get free API key at console.groq.com"
    )

    # Model selection
    model_name = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ],
        index=0
    )

# FIX: safe API key handling (Cloud + local)
api_key = os.getenv("GROQ_API_KEY") or api_key

# Clear button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize LLM
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
        (
            "system",
            "You are a helpful assistant powered by Groq. Answer clearly and concisely."
        ),
        ("user", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    return chain

# Get chain
chain = get_chain(api_key, model_name)

# If API missing
if not chain:
    st.warning("Please enter your Groq API key in the sidebar or Streamlit Secrets.")
    st.markdown("[Get your free API key here](https://console.groq.com/home)")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if question := st.chat_input("Ask me anything"):

    # USER message (FIXED)
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    # ASSISTANT response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in chain.stream({"question": question}):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # FIXED history saving
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Examples section
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
st.markdown("Built with LangChain & Groq 🚀")
