How to useCreate folder ./documents
Put your .pdf, .txt, .md files there
Run ollama serve (if not already running)

source rag_env/bin/activate
python local_rag_app.py

First run → indexes documents (takes longest).
Later runs → loads existing index (~2–5 seconds).
Open http://127.0.0.1:7860

