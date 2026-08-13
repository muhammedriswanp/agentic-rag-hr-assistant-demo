from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import time
from langchain_groq import ChatGroq


load_dotenv()

# ─── 1. Load & Split ───────────────────────────────────────────────
def load_documents():
    loader = TextLoader("data/hr_policy.txt", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# ─── 2. Build Vector Store ─────────────────────────────────────────
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="chroma_db/traditional"
    )
    return vectorstore

def traditional_rag(question, vectorstore, llm):
    start = time.time()
    # Single retrieval — no evaluation, no retry
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = PromptTemplate.from_template("""
You are an HR assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:""")

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    elapsed = time.time() - start
    return {
        "question": question,
        "answer": response.content,
        "retrieved_chunks": len(retrieved_docs),
        "retrievals": 1,
        "time_taken": round(elapsed, 2)
    }

# ─── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading documents...")
    chunks = load_documents()
    print(f"Total chunks: {len(chunks)}")

    print("Building vector store...")
    vectorstore = build_vectorstore(chunks)

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    test_questions = [
        "How many days of annual leave do employees get?",
        "What is the remote work policy?",
        "What happens if an employee breaches confidentiality?",
        "How is the performance bonus calculated and when is it paid?"
    ]

    print("\n" + "="*60)
    print("TRADITIONAL RAG RESULTS")
    print("="*60)

    results = []
    for q in test_questions:
        result = traditional_rag(q, vectorstore, llm)
        results.append(result)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"   [Retrievals: {result['retrievals']} | Chunks: {result['retrieved_chunks']} | Time: {result['time_taken']}s]")

    print("\n Traditional RAG complete.")



