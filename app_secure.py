import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import db

load_dotenv()
db.init_db()

st.set_page_config(page_title="Research Paper Assistant", page_icon="📄", layout="wide")

if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.username = None


# ================= LOGIN / REGISTER SCREEN =================
def show_login_page():
    st.title("📄 Research Paper Assistant")
    st.caption("Chat with and compare research papers — sign in to save your history.")
    login_tab, register_tab = st.tabs(["🔑 Login", "📝 Register"])

    with login_tab:
        st.subheader("Welcome back")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", use_container_width=True):
            if not username or not password:
                st.warning("Please enter both username and password.")
            else:
                user_id = db.login_user(username, password)
                if user_id:
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

    with register_tab:
        st.subheader("Create an account")
        new_user = st.text_input("Choose a username", key="reg_user")
        new_pass = st.text_input("Choose a password", type="password", key="reg_pass")
        confirm = st.text_input("Confirm password", type="password", key="reg_confirm")
        if st.button("Register", use_container_width=True):
            if not new_user or not new_pass:
                st.warning("Please fill in all fields.")
            elif new_pass != confirm:
                st.error("Passwords do not match.")
            elif len(new_pass) < 4:
                st.warning("Password should be at least 4 characters.")
            else:
                ok, message = db.register_user(new_user, new_pass)
                st.success(message) if ok else st.error(message)


# ================= THE MAIN CHAT APP =================
@st.cache_resource
def build_database(files_data):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for filename, file_bytes in files_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        for chunk in chunks:
            chunk.metadata["paper"] = filename
        all_chunks.extend(chunks)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(all_chunks, embeddings)


def is_comparison_request(llm, message):
    classify_prompt = f"""Decide if the user's message is asking to COMPARE
two or more papers (including which is better, which to choose, how they differ,
advantages of one over another) or if it is a NORMAL single question.
Reply with only one word: COMPARE or NORMAL.
User message: "{message}" """
    result = llm.invoke(classify_prompt).content.strip().upper()
    return "COMPARE" in result


def show_main_app():
    user_id = st.session_state.user_id
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

    # ---------- SIDEBAR: profile + history ----------
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        st.caption("Logged in")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()

        st.divider()
        st.markdown("### 🕘 Your History")

        # Load this user's saved messages from the database
        history = db.get_history(user_id)
        if not history:
            st.caption("No past conversations yet.")
        else:
            # Show only the user's past questions as a clickable-looking list
            for msg in history:
                if msg["role"] == "user":
                    st.markdown(f"- {msg['content'][:40]}...")

        st.divider()
        if st.button("🗑️ Clear My History", use_container_width=True):
            db.clear_history(user_id)
            st.rerun()

    # ---------- MAIN AREA: chat ----------
    st.title("📄 Research Paper Assistant")

    uploaded_files = st.file_uploader(
        "Upload your PDFs", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        files_data = tuple((f.name, f.getvalue()) for f in uploaded_files)
        paper_names = [f.name for f in uploaded_files]
        with st.spinner(f"Processing {len(uploaded_files)} paper(s)..."):
            db_vectors = build_database(files_data)
        st.success(f"{len(uploaded_files)} paper(s) ready!")

        # Show this user's saved history as chat bubbles
        for msg in db.get_history(user_id):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_message = st.chat_input("Ask a question, or ask me to compare the papers...")

        if user_message:
            with st.chat_message("user"):
                st.write(user_message)
            db.save_message(user_id, "user", user_message)   # SAVE to database

            is_compare = is_comparison_request(llm, user_message)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if is_compare and len(paper_names) >= 2:
                        per_paper_text = ""
                        for name in paper_names:
                            results = db_vectors.similarity_search(
                                user_message, k=4, filter={"paper": name})
                            chunks_text = "\n".join(r.page_content for r in results)
                            per_paper_text += f"\n\n=== PAPER: {name} ===\n{chunks_text}"
                        prompt = f"""You are comparing research papers.
The user asked: {user_message}
For each paper, briefly state its approach, then give a clear point-by-point
comparison of similarities and differences. End with a short summary.
{per_paper_text}"""
                    else:
                        relevant = db_vectors.similarity_search(user_message, k=5)
                        context = "\n\n".join(c.page_content for c in relevant)
                        prompt = f"""Answer using ONLY the context below.
Context:
{context}
Question: {user_message}"""

                    answer = llm.invoke(prompt).content
                    st.write(answer)
            db.save_message(user_id, "assistant", answer)   # SAVE to database
    else:
        st.info("👆 Upload one or more PDFs to start chatting.")


# ================= DECIDE WHAT TO SHOW =================
if st.session_state.user_id is None:
    show_login_page()
else:
    show_main_app()