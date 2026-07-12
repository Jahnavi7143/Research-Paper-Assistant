import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

# Paste your Groq key here between the quotes
load_dotenv()

# 1. Load PDF
loader = PyPDFLoader("paper.pdf")
pages = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)
print("Chunks:", len(chunks))

# 3. Make chunks searchable
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(chunks, embeddings)

# 4. Connect the AI (the model that writes answers)
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# 5. Build the RAG: find relevant chunks, then let the AI answer using them
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(search_kwargs={"k": 4}))

# 6. Ask a question
question = "What problem does this paper solve and what method do they use?"
answer = qa.invoke(question)
print("\nQUESTION:", question)
print("\nANSWER:", answer["result"])