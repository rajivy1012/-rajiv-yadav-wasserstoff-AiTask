
import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Configuring page settings
st.set_page_config(
    page_title="AI Website Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Applying custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .css-1d391kg {
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Configuring API Key
GOOGLE_API_KEY = "AIzaSyBR_xQQ7VuuDNNt1r7i2vXvo2_zIZCFEUo"  # Google Gemini API key
genai.configure(api_key=GOOGLE_API_KEY)

def fetch_webpage_text(url):
    """
    Fetches and extracts text content from a given URL.

    Args:
        url (str): The URL to fetch content from

    Returns:
        str: Extracted text content from the webpage
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from paragraphs and headers
        text = ' '.join([elem.get_text() for elem in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
        return text
    except Exception as e:
        st.error(f"Error fetching webpage: {str(e)}")
        return ""

def get_text_chunks(text):
    """
    Splits the input text into manageable chunks for processing.

    Args:
        text (str): Input text to be split

    Returns:
        list: List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """
    Creates and saves a FAISS vector store from text chunks.

    Args:
        text_chunks (list): List of text chunks to be vectorized
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """
    Creates a conversation chain with the language model.

    Returns:
        Chain: A configured QA chain
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, respond with 
    "I apologize, but I couldn't find the answer to your question in the website content."
    Please maintain a professional and helpful tone.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    """
    Processes user input and generates a response.

    Args:
        user_question (str): User's input question
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    with st.spinner("Thinking..."):
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        # Create a nice looking response container
        st.markdown("### Answer")
        st.markdown(response["output_text"])
        st.divider()

# Main application layout
st.title(" 🤖 Website Content Chatbot")
st.markdown("""
    This chatbot can answer questions about any website's content. 
    Just enter a URL in the sidebar and start asking questions!
""")

# Sidebar configuration code
with st.sidebar:
    st.header("📑 Website Configuration")
    st.markdown("Enter the website URL you'd like to chat about.")

    url = st.text_input("Website URL", placeholder="https://example.com")

    process_button = st.button("📥 Load Website", use_container_width=True)

    if process_button and url:
        with st.spinner("📄 Reading website content..."):
            raw_text = fetch_webpage_text(url)
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                with st.spinner("🔄 Processing content..."):
                    get_vector_store(text_chunks)
                st.success("✅ Website content loaded successfully!")
                st.balloons()
            else:
                st.error("❌ Failed to load website content")

# Main chatbot interface
st.divider()
user_question = st.text_input("Ask a question about the website:", placeholder="What would you like to know?")

if user_question:
    if not os.path.exists("faiss_index"):
        st.warning("🔍 Please load a website first using the sidebar!")
    else:
        user_input(user_question)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with Streamlit, LangChain, and Google's Gemini </p>
    </div>
""", unsafe_allow_html=True)
