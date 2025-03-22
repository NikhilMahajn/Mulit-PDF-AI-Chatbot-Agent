import os
import streamlit as st
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables (like API keys)
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Load and process PDFs without writing to disk
def loadPDF(files):
    try:
        st.write("Processing PDFs...")
        documents = []
        i = 1
        # Load documents directly from uploaded files
        if not os.path.exists("pdf"):
            os.mkdir("pdf")
        for file in files:
            with open(f"pdf/temp{i}.pdf", "wb") as f:
                 f.write(file.getvalue())
            i+=1                
        loader = PyPDFDirectoryLoader("pdf")
        documents.extend(loader.load())

        st.write(f"Loaded {len(documents)} pages from PDFs.")

        # Split documents into smaller chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        split_documents = splitter.split_documents(documents)

        # Create FAISS vector store and save locally
        st.write("Storing vectors in FAISS database...")
        vectors = FAISS.from_documents(split_documents, HuggingFaceEmbeddings())
        vectors.save_local("vectordb")
        st.success("VectorDB successfully created and saved!")
        return True
    except Exception as e:
        st.error(f"Error during PDF processing: {e}")
        return False
    finally:
        for filename in os.listdir("pdf"):
            file_path = os.path.join("pdf", filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)


def get_chain():
    # Check if the vectorstore already exists
    vector_retriever = FAISS.load_local("vectordb", embeddings=HuggingFaceEmbeddings(), allow_dangerous_deserialization=True).as_retriever()

    # Define the chat prompt
    chat_prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful AI assistant with deep expertise in the contents of the uploaded PDF files.
    Use the context extracted from the PDF documents to answer questions accurately.

    Rules:
    - ONLY use the context provided in the PDFs. Do not make up information.
    - Provide concise answers when the question is direct.
    - When appropriate, offer additional insights based on the provided context.
    - If the answer isn't found in the documents, respond with: "I'm sorry, I couldn't find the answer in the uploaded documents."

    Context:
    <context>
    {context}
    </context>

    Question: {input}

    Answer:
    """
)


    # Initialize the LLM and chain
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    doc_chain = create_stuff_documents_chain(llm, chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vector_retriever, combine_docs_chain=doc_chain)
    return retrieval_chain





# Streamlit UI
st.title("Multi-PDF AI ChatBot Agent")
st.sidebar.image("image.png")
st.sidebar.title("📁Upload Files Here")


uploaded_files = st.sidebar.file_uploader("", type="pdf", accept_multiple_files=True)
if st.sidebar.button("Upload & Process"):
    if uploaded_files:
        if loadPDF(uploaded_files):
            st.session_state["chain"] = get_chain()  # Save chain in session state for reusability
            st.success("AI ChatBot is ready! You can now ask questions.")

if "chain" in st.session_state:
    chain = st.session_state["chain"]
    question = st.text_input("Ask your question:")
    if question:
        with st.spinner("Thinking..."):
            ans = chain.invoke({'input': question})
            st.write(ans['answer'])
st.sidebar.write("Project By [Nikhil Mahajan](https://github.com/NikhilMahajn)")

import streamlit as st

