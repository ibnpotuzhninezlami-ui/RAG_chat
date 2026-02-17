"""
Simple fully-local RAG chat with PDF/TXT support – using FAISS instead of Chroma
Ollama + LangChain + Gradio + FAISS
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import gradio as gr


# ─── Configuration ────────────────────────────────────────────────────────

CONFIG = {
    "embedding_model": "nomic-embed-text",
    "llm_model": "llama3.2:3b",               # change to qwen2.5:7b, gemma2:9b, etc.
    "ollama_base_url": "http://localhost:11434",
    "index_path": "./faiss_index",            # folder where FAISS index will be saved
    "documents_folder": "./documents",        # put your PDFs & .txt files here
    "chunk_size": 850,
    "chunk_overlap": 140,
    "retriever_k": 5,
    "temperature": 0.25,
}

PROMPT_TEMPLATE = """\
Використовуй **тільки** інформацію з наданого контексту для відповіді.


Контекст:
{context}

Питання: {question}

Відповідь (коротко, чітко):\
"""


# ─── Helper functions ─────────────────────────────────────────────────────

def get_document_loader(file_path: Path):
    """Select loader based on file extension."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(file_path))
    if ext in (".txt", ".md"):
        return TextLoader(str(file_path), encoding="utf-8")
    return None


def load_all_documents(folder_path: str) -> List:
    """Load PDFs + txt/md from folder."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    docs = []
    for file_path in folder.rglob("*"):
        if file_path.is_file():
            loader = get_document_loader(file_path)
            if loader:
                try:
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"⚠️ Failed to load {file_path.name}: {e}")
    return docs


def build_faiss_index(documents_folder: str, index_path: str, embeddings) -> FAISS:
    """Create and save FAISS index from documents."""
    print("Loading documents...")
    raw_docs = load_all_documents(documents_folder)

    if not raw_docs:
        raise ValueError("No documents loaded. Check folder and file formats.")

    print(f"→ Loaded {len(raw_docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        keep_separator=True,
    )

    chunks = text_splitter.split_documents(raw_docs)
    print(f"→ Created {len(chunks)} chunks")

    print("Building FAISS index (may take a while first time)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"FAISS index saved to: {index_path}")
    return vectorstore


def get_or_create_vectorstore(embeddings) -> FAISS:
    """Load existing FAISS index or build new one."""
    index_dir = Path(CONFIG["index_path"])

    if index_dir.exists() and any(index_dir.iterdir()):
        print("Loading existing FAISS index...")
        return FAISS.load_local(
            CONFIG["index_path"],
            embeddings,
            allow_dangerous_deserialization=True  # Required since LangChain 0.2+
        )
    else:
        return build_faiss_index(
            documents_folder=CONFIG["documents_folder"],
            index_path=CONFIG["index_path"],
            embeddings=embeddings,
        )


# ─── RAG chain ────────────────────────────────────────────────────────────

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    embeddings = OllamaEmbeddings(
        model=CONFIG["embedding_model"],
        base_url=CONFIG["ollama_base_url"],
    )

    vectorstore = get_or_create_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": CONFIG["retriever_k"]})

    llm = OllamaLLM(
        model=CONFIG["llm_model"],
        base_url=CONFIG["ollama_base_url"],
        temperature=CONFIG["temperature"],
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ─── Gradio interface ─────────────────────────────────────────────────────

def chat_with_rag(message: str, history):
    try:
        answer = rag_chain.invoke(message)
        return answer
    except Exception as e:
        return f"Помилка: {str(e)}"


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting local RAG application with FAISS...")

    # Build/load once
    rag_chain = build_rag_chain()

    demo = gr.ChatInterface(
        fn=chat_with_rag,
        title="Локальний RAG • PDF + TXT • FAISS + Ollama",
        description=(
            f"Документи з папки: {CONFIG['documents_folder']}\n"
            f"Модель: {CONFIG['llm_model']} • Ембедінг: {CONFIG['embedding_model']}\n"
            "Індекс зберігається в: ./faiss_index"
        ),
        examples=[
            "History of Person Centered Therapy",
            "Solution Focused Therap Technniques",
            "Cognitive Behaviuor Thearapy History",
        ],
        cache_examples=False,
    )

    demo.launch(share=False)
