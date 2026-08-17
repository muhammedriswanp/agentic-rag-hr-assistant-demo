from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import time

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
        persist_directory="chroma_db/agentic"
    )
    return vectorstore

# ─── 3. Evaluate: Is context sufficient? ───────────────────────────
def evaluate_context(question, context, llm):
    prompt = PromptTemplate.from_template("""
You are evaluating whether a retrieved context is sufficient to answer a question.

Question: {question}
Retrieved Context: {context}

Is the context sufficient to give a complete and accurate answer?
Reply with ONLY one word: YES or NO.
""")
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})
    return response.content.strip().upper().startswith("YES")

# ─── 4. Reformulate Query ──────────────────────────────────────────
def reformulate_query(question, context, llm):
    prompt = PromptTemplate.from_template("""
The following question could not be answered with the retrieved context.
Reformulate the question to search for more specific information.

Original Question: {question}
Retrieved Context (insufficient): {context}

Write a better search query (one line only, no explanation):
""")
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})
    return response.content.strip()

# ─── 5. Generate Final Answer ──────────────────────────────────────
def generate_answer(question, context, llm):
    prompt = PromptTemplate.from_template("""
You are an HR assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:""")
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content

def agentic_rag(question, vectorstore, llm, max_attempts=3):
    start = time.time()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    current_query = question
    total_retrievals = 0
    all_context = ""
    retrieval_log = []

    for attempt in range(1, max_attempts + 1):
        print(f"   [Attempt {attempt}] Query: {current_query}")

        # Retrieve
        retrieved_docs = retriever.invoke(current_query)
        total_retrievals += 1
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Merge context across attempts
        all_context = context if attempt == 1 else all_context + "\n\n" + context

        retrieval_log.append({
            "attempt": attempt,
            "query": current_query,
            "chunks": len(retrieved_docs)
        })

        # Evaluate sufficiency
        is_sufficient = evaluate_context(question, all_context, llm)
        print(f"   [Attempt {attempt}] Context sufficient: {is_sufficient}")

        if is_sufficient:
            break

        # Reformulate if not last attempt
        if attempt < max_attempts:
            current_query = reformulate_query(question, all_context, llm)

    # Generate final answer with all collected context
    answer = generate_answer(question, all_context, llm)
    elapsed = time.time() - start

    return {
        "question": question,
        "answer": answer,
        "total_retrievals": total_retrievals,
        "retrieval_log": retrieval_log,
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
        "How many days of annual leave do employees get?",           # easy — resolves in 1
        "What is the remote work policy?",                           # easy — resolves in 1
        "What happens if an employee breaches confidentiality?",     # easy — resolves in 1
        "How is the performance bonus calculated and when is it paid?", # easy — resolves in 1
        "Can a new joiner work from home on their first week?",      # tricky — needs reformulation
        "What are the financial penalties for gift policy violations?", # tricky — needs reformulation
    ]

    print("\n" + "="*60)
    print("AGENTIC RAG RESULTS")
    print("="*60)

    results = []
    for q in test_questions:
        print(f"\nQ: {q}")
        result = agentic_rag(q, vectorstore, llm)
        results.append(result)
        print(f"A: {result['answer']}")
        print(f"   [Total Retrievals: {result['total_retrievals']} | Time: {result['time_taken']}s]")

    print("\n Agentic RAG complete.")
    