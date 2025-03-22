# Multi-PDF Retrieval-Augmented Generation (RAG) System

This project implements a multi-PDF retrieval-augmented generation (RAG) chatbot system that allows users to upload multiple PDF files, process them into embeddings, and ask questions based on the content of the uploaded PDFs. The system leverages FAISS, LangChain, and Streamlit to create a seamless user experience for document-based question answering.

---

## Features

- **Multiple PDF File Upload**: Upload and process multiple PDF documents simultaneously.
- **Document Chunking**: Splits PDF content into manageable chunks to enhance retrieval accuracy.
- **FAISS Vector Database**: Stores document embeddings locally using FAISS for fast and efficient retrieval.
- **Generative AI Chatbot**: Answers user queries based on retrieved document contexts.
- **Customizable Prompt**: Tailored prompt setup to specialize the chatbot on specific knowledge domains (e.g., history of Chhatrapati Shivaji Maharaj).

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root and add your GROQ API key:
     ```bash
     GROQ_API_KEY=your_groq_api_key_here
     ```

---

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Upload PDF files using the sidebar file uploader.
3. Click the "Upload & Process" button to process and store embeddings.
4. Ask questions based on the uploaded PDF content through the chatbot interface.

---

## Project Structure

```
.
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
├── pdf/                     # Temporary storage for uploaded PDFs
└── vectordb/                # FAISS vector database storage
```

---

## Key Libraries Used

- **LangChain**: For text splitting, document chains, and LLM integration.
- **FAISS**: Fast Approximate Nearest Neighbor Search for embedding storage and retrieval.
- **Streamlit**: For building the interactive UI.
- **PyPDFLoader**: Efficient PDF document loading.

---

## Custom Chat Prompt

The chatbot uses a specialized prompt designed for Maratha Samrajya history, particularly Chhatrapati Shivaji Maharaj. This ensures that the system answers questions contextually and accurately within this knowledge domain.

```text
You are a helpful assistant who is an expert in Maratha Samrajya History and knows everything about Chhatrapati Shivaji Maharaj.
Answer the question accordingly with respect to context only:

<context>
{context}
</context>

Question: {input}
```

---

## Future Enhancements

- Add support for different file types (e.g., Word documents, images).
- Improve chatbot generalization for broader domains.
- Implement more advanced vector search techniques.

---

## Contributing

Feel free to submit issues, fork the repository, and make pull requests. Contributions are welcome!

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

---

## Acknowledgements

- Inspired by LangChain's document-based RAG examples.
- Thanks to the Streamlit and FAISS communities for their excellent libraries.
