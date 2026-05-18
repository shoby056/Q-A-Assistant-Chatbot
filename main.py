import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load local env (for local run only)
load_dotenv()

# Page config
st.set_page_config(
    page_title="Simple LangChain Chatbot With Groq",
    page_icon="🚀"
)

# Title
st.title("🚀 Simple LangChain Chatbot With Groq")
st.markdown("Learn LangChain basics with Groq's ultra-fast inference")

# Get API key (Streamlit Cloud + Local support)
api_key = os.getenv("GROQ_API_KEY")

# Sidebar
with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ],
        index=0
    )

# Clear chat
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize LLM chain
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

    chain = prompt | llm | StrOutputParser()

    return chain

chain = get_chain(api_key, model_name)

# If API key missing
if not api_key:
    st.warning("⚠️ Please add your GROQ_API_KEY in Streamlit Secrets or .env file")
    st.stop()

# Chat UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if question := st.chat_input("Ask me anything"):

    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    # AI response
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
            st.error(f"Error: {str(e)}")



##examples
st.markdown("---")
st.markdown("### 💡 Try these Examples:")
col1,col2=st.columns(2)
with col1:
    st.markdown("- What is Langchain?")
    st.markdown("- Explain Groq's Lpu technology")

with col2:
    st.markdown("- How do i Learn Programming?")
    st.markdown("- Write a haiku about Ai")

#Footer
st.markdown("---")
st.markdown("Built with Langchain & Groq | Experience the speed! ")
