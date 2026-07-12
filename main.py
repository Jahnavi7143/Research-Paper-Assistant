import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# Load the Groq API key from the .env file
load_dotenv()

# ---------- Page setup ----------
st.title("📄 Research Paper Assistant")
st.write("Upload research papers, then ask questions or ask me to compare them — all in one chat.")

# ---------- Upload multiple PDFs ----------
uploaded_files = st.file_uploader(
    "Upload your PDFs", type="pdf", accept_multiple_files=True
)


# ---------- Build the searchable database from the uploaded papers ----------
@st.cache_resource
def build_database(files_data):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for filename, file_bytes in files_data:
        # Save the file temporarily so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        chunks = splitter.split_documents(pages)

        # Tag every chunk with the paper it came from (used for comparison)
        for chunk in chunks:
            chunk.metadata["paper"] = filename

        all_chunks.extend(chunks)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(all_chunks, embeddings)
    return db


# The AI model that writes answers
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)


# ---------- Main app ----------
if uploaded_files:
    files_data = tuple((f.name, f.getvalue()) for f in uploaded_files)
    paper_names = [f.name for f in uploaded_files]

    with st.spinner(f"Processing {len(uploaded_files)} paper(s)..."):
        db = build_database(files_data)
    st.success(
        f"{len(uploaded_files)} paper(s) ready! Ask a question or ask me to compare the papers."
    )

    # One single history for everything (questions and comparisons together)
    if "history" not in st.session_state:
        st.session_state.history = []

    # Show the full conversation so far
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_message = st.chat_input("Ask a question, or ask me to compare the papers...")

    if user_message:
        # Show + save the user's message
        with st.chat_message("user"):
            st.write(user_message)
        st.session_state.history.append({"role": "user", "content": user_message})

        # Let the LLM decide whether this is a comparison request (understands meaning,
        # so it catches "which one is better?", "which should I use?", etc.)
        def is_comparison_request(message):
            classify_prompt = f"""Decide if the user's message is asking to COMPARE
two or more papers (this includes asking which is better, which to choose,
how one differs from another, advantages of one over another, etc.)
or if it is a NORMAL single question.

Reply with only one word: COMPARE or NORMAL.

User message: "{message}"
"""
            result = llm.invoke(classify_prompt).content.strip().upper()
            return "COMPARE" in result

        is_compare = is_comparison_request(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                # ---- COMPARISON path: retrieve from each paper separately ----
                if is_compare and len(paper_names) >= 2:
                    per_paper_text = ""
                    for name in paper_names:
                        results = db.similarity_search(
                            user_message, k=4, filter={"paper": name}
                        )
                        chunks_text = "\n".join(r.page_content for r in results)
                        per_paper_text += f"\n\n=== PAPER: {name} ===\n{chunks_text}"

                    prompt = f"""You are comparing research papers.
The user asked: {user_message}

For each paper, briefly state its approach, then give a clear point-by-point
comparison highlighting similarities and differences. End with a short summary.

{per_paper_text}
"""

                # ---- NORMAL path: retrieve across all papers ----
                else:
                    relevant = db.similarity_search(user_message, k=5)
                    context = "\n\n".join(c.page_content for c in relevant)
                    prompt = f"""Answer using ONLY the context below.

Context:
{context}

Question: {user_message}
"""

                answer = llm.invoke(prompt).content
                st.write(answer)

        # Save the answer so it stays in the chat history
        st.session_state.history.append({"role": "assistant", "content": answer})