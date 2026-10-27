# DocChat

Chat with your own documents. Add PDFs, Word files, slides, spreadsheets, CSV, HTML, text, images (OCR) or whole websites, then ask questions. Every answer cites the exact passages it used. Python + Streamlit, free tools only.

## Features
- Reads PDF (scanned pages via OCR), DOCX, PPTX, XLSX, CSV, HTML, TXT/MD/JSON and images, plus web pages and multi-page website crawling
- Hybrid search: BM25 keywords + optional semantic embeddings (English or multilingual), merged with Reciprocal Rank Fusion
- Streaming answers with citations from Gemini (free key), any OpenAI-compatible API (Groq, OpenRouter) or local Ollama; works with no AI too (extractive answers)
- Follow-up questions, answer style and reply language, custom instructions, quick actions (summary, key points, FAQ, suggested questions)
- Source cards with highlighted matches, thumbs up/down, chat export, insights dashboard
- Evaluation page: retrieval hit rate and MRR on your own test questions
- Safety: prompt-injection guard, private-network URLs blocked, nothing stored on the server
- Automated tests: `python test_system.py`

## Speed
Gemini is tuned for speed: thinking is switched off, the upload is shrunk before sending, one connection is reused, the last working model is remembered, and a retired or rate-limited model is skipped automatically. Choose **Fast** (Flash-Lite models, default) or **Accurate** (full Flash) in the sidebar. Identical questions are answered instantly from a cache, and the page shows the time to first words for every answer.

## Gemini key (free)
Never put the key inside the code or upload it to GitHub. Choose one:
- On your PC: `python setup.py --key` (saves it to `.streamlit/secrets.toml`; keep that file on your computer only)
- On Streamlit Cloud: app Settings > Secrets > `GEMINI_API_KEY = "your-key"`
- Or paste it into the sidebar under AI model

The default engine becomes Gemini when a key is present, and the model is chosen automatically.

## Run
```bash
python setup.py --run
```
Later: `streamlit run app.py`.

## Files (one folder)
`app.py` interface · `readers.py` files and websites · `index.py` chunking and search · `llm.py` AI clients · `rag.py` prompts and evaluation · `test_system.py` tests · `setup.py` installer · `requirements.txt`, `packages.txt`

## Deploy free
Upload all files to a public GitHub repo, create the app at https://share.streamlit.io with main file `app.py`, and pick Python 3.11. Optional secrets (Settings > Secrets): `GEMINI_API_KEY` or `LLM_API_KEY`.
