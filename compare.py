from agentic_rag_chroma import agentic_rag, load_documents, build_vectorstore
from agentic_rag_pinecone import (
    agentic_rag_pinecone,
    load_documents as load_docs_pinecone,
    build_vectorstore as build_pinecone
)
from langchain_groq import ChatGroq

TEST_QUESTIONS = [
    "How many days of annual leave do employees get?",
    "What is the remote work policy?",
    "What happens if an employee breaches confidentiality?",
    "How is the performance bonus calculated and when is it paid?",
    "Can a new joiner work from home on their first week?",
    "What are the financial penalties for gift policy violations?",
]

def run_comparison():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    print("Setting up ChromaDB...")
    chroma_chunks = load_documents()
    chroma_vectorstore = build_vectorstore(chroma_chunks)

    print("Setting up Pinecone...")
    pinecone_chunks = load_docs_pinecone()
    pinecone_vectorstore = build_pinecone(pinecone_chunks)

    print("\n" + "="*70)
    print("COMPARISON: Agentic RAG — ChromaDB vs Pinecone")
    print("="*70)

    chroma_times, pinecone_times = [], []

    for q in TEST_QUESTIONS:
        print(f"\nQ: {q}")

        ChromaDB_Result = agentic_rag(q, chroma_vectorstore, llm)
        Pinecone_Result = agentic_rag_pinecone(q, pinecone_vectorstore, llm)

        chroma_times.append(ChromaDB_Result["time_taken"])
        pinecone_times.append(Pinecone_Result["time_taken"])

        print(f"  [ChromaDB]  Time: {ChromaDB_Result['time_taken']}s | Retrievals: {ChromaDB_Result['total_retrievals']}")
        print(f"  [Pinecone]  Time: {Pinecone_Result['time_taken']}s | Retrievals: {Pinecone_Result['retrievals']}")
        print(f"  Chroma Answer : {ChromaDB_Result['answer'][:120]}...")
        print(f"  Pinecone Answer: {Pinecone_Result['answer'][:120]}...")

    print("\n" + "="*70)
    print("LATENCY REPORT")
    print("="*70)
    print(f"  ChromaDB  avg time : {round(sum(chroma_times)/len(chroma_times), 2)}s")
    print(f"  Pinecone  avg time : {round(sum(pinecone_times)/len(pinecone_times), 2)}s")
    faster = "Pinecone" if sum(pinecone_times) < sum(chroma_times) else "ChromaDB"
    print(f"   Faster overall  : {faster}")


if __name__ == "__main__":
    run_comparison()