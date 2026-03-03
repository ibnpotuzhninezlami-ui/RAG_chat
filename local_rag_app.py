"""
Fully local RAG chat with PDF/TXT support – using FAISS
Ollama + LangChain + Gradio + FAISS
Compatible with Gradio 6.0+ (messages format only)
"""

from pathlib import Path
from typing import List

import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda  # Added RunnableLambda import
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage


# ─── Configuration ────────────────────────────────────────────────────────

CONFIG = {
    "embedding_model": "nomic-embed-text:latest",
    "llm_model":       "qwen2.5:7b-instruct",  # or gemma2:9b, llama3.2:3b, phi4:14b, etc.
    "ollama_base_url": "http://localhost:11434",
    "index_path":      "./faiss_index",
    "documents_folder":"./documents",
    "chunk_size":      5000,
    "chunk_overlap":   2000,
    "retriever_k":     8,
    "temperature":     0.1,
}

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful, accurate assistant.
Answer using **only** the provided context.
If the information is not in the context — say clearly that you don't know.
Use markdown when helpful."""),
    MessagesPlaceholder("history"),
    ("human", """Context:
{context}

Question: {question}"""),
])


# ─── Document Loading & Indexing ──────────────────────────────────────────

def get_document_loader(file_path: Path):
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(file_path))
    if ext in (".txt", ".md"):
        return TextLoader(str(file_path), encoding="utf-8")
    return None


def load_all_documents(folder_path: str) -> List:
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {folder}")

    docs = []
    for file_path in folder.rglob("*"):
        if file_path.is_file():
            loader = get_document_loader(file_path)
            if loader:
                try:
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"Failed to load {file_path.name}: {e}")
    return docs


def build_faiss_index(documents_folder: str, index_path: str, embeddings) -> FAISS:
    print("Loading documents...")
    raw_docs = load_all_documents(documents_folder)
    if not raw_docs:
        raise ValueError("No documents loaded. Check folder & formats (.pdf/.txt/.md)")

    print(f"→ Loaded {len(raw_docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        keep_separator=True,
    )

    chunks = text_splitter.split_documents(raw_docs)
    print(f"→ Created {len(chunks)} chunks")

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"Index saved to {index_path}")
    return vectorstore


def get_or_create_vectorstore(embeddings) -> FAISS:
    index_dir = Path(CONFIG["index_path"])
    if index_dir.exists() and any(index_dir.iterdir()):
        print("Loading existing FAISS index...")
        return FAISS.load_local(
            CONFIG["index_path"], embeddings,
            allow_dangerous_deserialization=True
        )
    return build_faiss_index(
        CONFIG["documents_folder"], CONFIG["index_path"], embeddings
    )


# ─── RAG Chain ────────────────────────────────────────────────────────────

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    embeddings = OllamaEmbeddings(
        model=CONFIG["embedding_model"],
        base_url=CONFIG["ollama_base_url"],
    )

    vectorstore = get_or_create_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": CONFIG["retriever_k"]})

    llm = ChatOllama(
        model=CONFIG["llm_model"],
        base_url=CONFIG["ollama_base_url"],
        temperature=CONFIG["temperature"],
    )

    # Define context logic as explicit function → Runnable
    def get_context(inputs):
        question = inputs["question"]
        docs = retriever.invoke(question)
        return format_docs(docs)

    context_runnable = RunnableLambda(get_context)

    rag_chain = (
        RunnablePassthrough.assign(context=context_runnable)
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ─── Gradio Handlers ──────────────────────────────────────────────────────

def chat_with_rag(message: str, history: list):
    if not message or not message.strip():
        return "", history

    # Convert Gradio messages format → LangChain messages
    langchain_history = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user" and content:
            langchain_history.append(HumanMessage(content=content))
        elif role == "assistant" and content:
            langchain_history.append(AIMessage(content=content))

    try:
        answer = rag_chain.invoke({
            "question": message,
            "history": langchain_history
        })
        # Append new messages in Gradio messages format
        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer.strip()}
        ]
        return "", updated_history
    except Exception as e:
        import traceback
        print("Error during inference:")
        print(traceback.format_exc())
        error_msg = f"⚠️ Error: {str(e)}"
        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg}
        ]
        return "", updated_history


def rebuild_index():
    import shutil
    try:
        shutil.rmtree(CONFIG["index_path"], ignore_errors=True)
        global rag_chain
        rag_chain = build_rag_chain()
        return "Index rebuilt successfully!"
    except Exception as e:
        return f"Rebuild failed: {str(e)}"


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting local RAG chat (Ollama + FAISS + Gradio 6.0+)...")
    rag_chain = build_rag_chain()

    with gr.Blocks() as demo:
        gr.Markdown(
            f"# Local RAG Chat\n"
            f"**Model**: {CONFIG['llm_model']} • **Embeddings**: {CONFIG['embedding_model']}\n\n"
            f"Documents folder: `{CONFIG['documents_folder']}` • Index: `./faiss_index`"
        )

        chatbot = gr.Chatbot(
            height=600,
            show_label=False,
            # No 'type' parameter in Gradio 6.0+
        )
        msg = gr.Textbox(
            placeholder="Ask about your documents...",
            label="Question"
        )

        with gr.Row():
            submit = gr.Button("Send")
            clear = gr.Button("Clear")
            rebuild = gr.Button("Rebuild Index")

        status = gr.Markdown("Ready.")

        # Events
        msg.submit(chat_with_rag, [msg, chatbot], [msg, chatbot])
        submit.click(chat_with_rag, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: ("", []), None, [msg, chatbot], queue=False)
        rebuild.click(rebuild_index, None, status)

        gr.Examples(
            examples=[
                "What this text about",
                "Summarize the main ideas in the documents",
                "Compare different trading strategies from this documents",
                "Pros and cons of this trading strategies",
            ],
            inputs=msg,
        )

    demo.launch(share=False)
