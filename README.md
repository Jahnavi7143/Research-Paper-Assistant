# 📄 Research Paper Assistant

An AI-powered **RAG (Retrieval-Augmented Generation)** application that lets users **chat with research papers**, **compare multiple papers side by side**, and keep a **private, persistent history** of their conversations behind a secure login.

> Built while reviewing literature for a building-damage-assessment project, to solve a real problem: understanding dense research papers and comparing their methods, datasets, and results *quickly* — without re-reading each one end to end, and without the answers drifting away from what the papers actually say.

---

## ✨ Features

- **💬 Chat with papers** — ask questions in plain language and get answers grounded strictly in the uploaded PDFs (not made-up).
- **📚 Multi-paper support** — upload several papers at once and query across all of them.
- **⚖️ Smart comparison** — clear, point-by-point comparisons of methods, datasets, results, and limitations.
- **🧠 Automatic intent routing** — an LLM classifies each message as a *question* or a *comparison*, so even phrasings like *"which one is better?"* are handled correctly (no rigid keyword matching).
- **🎯 Balanced per-paper retrieval** — for comparisons, relevant sections are retrieved from **each paper separately**, so every paper is represented fairly instead of one dominating.
- **🔐 User accounts** — register/login with securely **hashed passwords** (bcrypt); each user's data is private.
- **🕘 Persistent, per-user history** — conversations are saved to a database and reloaded on the next login, shown in a sidebar.

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| Language | **Python** |
| RAG orchestration | **LangChain** |
| Embeddings | **HuggingFace — all-MiniLM-L6-v2** (runs locally, free) |
| Vector database | **Chroma** |
| LLM (answer generation) | **Groq — openai/gpt-oss-20b** |
| Web interface | **Streamlit** |
| App database (users + history) | **SQLite** |
| Password security | **bcrypt** |
| Config / secrets | **python-dotenv** |

---

## ⚙️ How It Works

1. **Authenticate** — users register or log in; passwords are hashed with bcrypt and stored in SQLite.
2. **Load** — uploaded PDFs are read and their text is extracted.
3. **Chunk** — text is split into overlapping chunks (1000 chars, 200 overlap) so no context is lost at the edges.
4. **Embed & store** — each chunk is embedded and stored in a Chroma vector database, tagged with its source paper.
5. **Route** — an LLM classifies the user's message as a normal question or a comparison request.
6. **Retrieve & answer:**
   - *Normal question* → most relevant chunks across all papers are retrieved and passed to the LLM.
   - *Comparison* → relevant chunks are retrieved **per paper**, labeled by source, and the LLM produces a balanced comparison.
7. **Persist** — every message is saved to SQLite under the user's ID and reloaded on their next login.

\`\`\`
          Register / Login (bcrypt + SQLite)
                        │
PDF(s) ─▶ Chunk ─▶ Embed ─▶ Chroma vector DB
                                │
        User message ─▶ Intent classifier (LLM)
                                │
                ┌───────────────┴───────────────┐
          Normal question                  Comparison
        (retrieve across all)        (retrieve per paper, labeled)
                └───────────────┬───────────────┘
                                ▼
                          Groq LLM ─▶ Answer ─▶ saved to SQLite (per user)
\`\`\`

---

## 🚀 Setup & Run

**1. Clone the repository**
\`\`\`
git clone https://github.com/Jahnavi7143/Research-Paper-Assistant.git
cd Research-Paper-Assistant
\`\`\`

**2. (Recommended) Create and activate a virtual environment**
\`\`\`
python -m venv venv
venv\Scripts\activate      # Windows
\`\`\`

**3. Install dependencies**
\`\`\`
pip install -r requirements.txt
\`\`\`

**4. Add your Groq API key**

Create a file named \`.env\` in the project root:
\`\`\`
GROQ_API_KEY=your_key_here
\`\`\`
Get a free key at [console.groq.com](https://console.groq.com).

**5. Run the app**
\`\`\`
streamlit run app_secure.py
\`\`\`

Register an account, log in, upload one or more PDFs, then ask a question or ask it to compare the papers. Your history is saved automatically.

> \`main.py\` is the earlier single-session version (no login). \`app_secure.py\` is the full version with authentication and persistent history.

---

## 🔒 Security & Privacy Notes

- Passwords are **hashed with bcrypt** — never stored in plain text.
- Database queries use **parameterized statements** to prevent SQL injection.
- Secrets are kept in \`.env\` and excluded from version control via \`.gitignore\`.
- Each user's chat history is **isolated** to their own account.
- *Honest limitation:* answer generation calls a hosted LLM (Groq), so text does leave the machine at that step. A fully private deployment would use a local LLM.

---

## 🔮 Possible Future Improvements

- **Source citations** — show which paper and page each answer came from.
- **Fully private mode** — swap Groq for a local LLM (e.g. Ollama) so no data leaves the environment.
- **Separate conversations** — ChatGPT-style multiple named chats per user.
- **Answer evaluation** — measure accuracy on a fixed question set.
- **Production hardening** — HTTPS, rate limiting, stronger password/session policies.
- **Scale-up** — migrate from SQLite to PostgreSQL for multi-user, concurrent deployment.

---

## 📌 Notes

- The \`.env\` (API key) and \`app_data.db\` (user data) files are intentionally excluded from version control — never commit secrets or user data.
- The Chroma vector store is rebuilt per session from the uploaded PDFs; only user accounts and chat history are stored permanently (in SQLite).
