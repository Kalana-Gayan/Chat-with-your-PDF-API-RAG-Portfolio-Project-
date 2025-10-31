import os
import tempfile
import shutil
import uuid # <-- NEW IMPORT for manual ID generation
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from google.api_core import exceptions as google_exceptions

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document # <-- NEW IMPORT
from langchain_chroma import Chroma

# --- Application Setup ---

# Check for Google Service Account credentials
if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    print("INFO:     Found GOOGLE_APPLICATION_CREDENTIALS. Service account will be used for authentication.")
else:
    print("WARNING:  GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
    print("WARNING:  Please set it to the path of your service account JSON file.")
    print("WARNING:  Example: export GOOGLE_APPLICATION_CREDENTIALS=\"/path/to/your/confidential.json\"")


# Temporary directory to store uploaded files
TEMP_DIR = tempfile.mkdtemp()

app = FastAPI(
    title="Chat with your PDF API",
    description="An API to upload a PDF and ask questions about its content using LangChain and Gemini.",
    version="1.2-gemini-fix" # Version bump
)

# Global variables to hold the vector store and RAG chain
vector_store: Chroma | None = None
rag_chain = None

# --- Pydantic Models for API ---

class AskRequest(BaseModel):
    """Request model for the /ask endpoint"""
    question: str

# --- Helper Functions ---

def load_and_split_pdf(file_path: str) -> list:
    """Loads a PDF and splits it into text chunks."""
    print(f"Loading and splitting document: {file_path}")
    try:
        # 1. Load the document
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # 2. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(pages)
        print(f"Successfully split document into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"Error in load_and_split_pdf: {e}")
        raise

def create_vector_store(chunks: list[Document]) -> Chroma:
    """
    Creates an in-memory vector store from text chunks using a robust,
    manual method to bypass LangChain wrapper bugs.
    """
    print("Creating vector store...")
    try:
        # 1. Initialize the embeddings model
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # 2. Initialize an empty Chroma vector store
        # We use a temporary, in-memory client.
        # This wrapper is just to hold the in-memory client and embedding function
        db = Chroma(
            collection_name=f"pdf_chat_{uuid.uuid4()}", # Unique name for in-memory
            embedding_function=embeddings_model
        )

        # --- SURGICAL FIX ---
        # 3. Prepare data for the underlying chromadb client
        
        # 3a. Get the text content from each LangChain Document
        documents = [chunk.page_content for chunk in chunks]
        
        # 3b. Manually embed all documents in one batch
        print("Embedding documents...")
        embeddings = embeddings_model.embed_documents(documents)
        print("Embeddings created successfully.")

        # 3c. *** FINAL FIX: Force cast embeddings to simple lists ***
        embeddings = [list(e) for e in embeddings]

        # 3d. Generate unique IDs for each chunk
        ids = [str(uuid.uuid4()) for _ in documents]

        # 3e. Get the metadata from each LangChain Document
        metadatas = [chunk.metadata for chunk in chunks]

        # 4. Use the underlying ._collection.add() to insert data directly
        # This bypasses the buggy LangChain data parsing
        db._collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        # --- END OF FIX ---
        
        print("Vector store created and populated successfully.")
        return db
    except Exception as e:
        print(f"Error in create_vector_store: {e}")
        raise

def create_rag_chain(db: Chroma):
    """Creates a RAG (Retrieval-Augmented Generation) chain."""
    print("Creating RAG chain...")
    try:
        # 4. Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0  # We want factual answers
        )

        # 5. Create the prompt template
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Keep the answer concise and based *only* on the provided context."
            "\n\n"
            "<context>"
            "{context}"
            "</context>"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # 6. Create the "stuff documents" chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        # 7. Create the retriever
        retriever = db.as_retriever()

        # 8. Create the final retrieval chain
        final_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("RAG chain created successfully.")
        return final_chain
    except Exception as e:
        print(f"Error in create_rag_chain: {e}")
        raise

# --- API Endpoints ---

@app.get("/")
async def root():
    """Root endpoint to check if the server is running."""
    return {
        "message": "Welcome to the Chat with your PDF API!",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF, processes it, and creates the RAG chain."""
    global vector_store, rag_chain

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")

    file_path = ""
    try:
        # Save the uploaded file to our temporary directory
        safe_filename = file.filename.replace("..", "_").replace("/", "_")
        file_path = os.path.join(TEMP_DIR, safe_filename)
        
        # Use shutil.copyfileobj to stream the file to disk
        print(f"Saving uploaded file to: {file_path}")
        with open(file_path, "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)
        
        # Verify file size
        file_size = os.path.getsize(file_path)
        print(f"File saved successfully. Size: {file_size} bytes.")
        if file_size == 0:
            print("WARNING: The uploaded file is empty.")
            raise HTTPException(status_code=400, detail="The uploaded PDF file is empty.")

        # --- This is our main processing pipeline ---
        chunks = load_and_split_pdf(file_path)
        vector_store = create_vector_store(chunks)
        rag_chain = create_rag_chain(vector_store)
        # --- End of pipeline ---

        return {"message": f"Successfully processed '{file.filename}'. Ready to answer questions."}

    except google_exceptions.ResourceExhausted as e:
        print(f"Google API quota error: {e}")
        raise HTTPException(status_code=400, detail=f"Google API quota exceeded. Please check your plan. Error: {e.message}")
    except Exception as e:
        print(f"Error during upload: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        # Clean up the temporary file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary file: {file_path}")

@app.post("/ask")
async def ask_question(request: AskRequest):
    """Asks a question to the processed document."""
    if rag_chain is None:
        raise HTTPException(status_code=400, detail="No document has been processed. Please upload a PDF to /upload first.")

    try:
        # Use the RAG chain to get an answer
        response = rag_chain.invoke({"input": request.question})

        # The response is a dictionary, we just want the 'answer'
        return {"answer": response.get("answer", "No answer found.")}

    except google_exceptions.ResourceExhausted as e:
        print(f"Google API quota error during ask: {e}")
        raise HTTPException(status_code=400, detail=f"Google API quota exceeded. Please check your plan. Error: {e.message}")
    except Exception as e:
        print(f"Error during ask: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


