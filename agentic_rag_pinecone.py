from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
import time

INDEX_NAME = "hr-policy"

def load_documents():
    loader = TextLoader("data/hr_policy.txt", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    return vectorstore

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

def agentic_rag_pinecone(question, vectorstore, llm, max_attempts=3):
    start = time.time()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    current_query = question
    total_retrievals = 0
    all_context = ""

    for attempt in range(1, max_attempts + 1):
        retrieved_docs = retriever.invoke(current_query)
        total_retrievals += 1
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        all_context = context if attempt == 1 else all_context + "\n\n" + context

        is_sufficient = evaluate_context(question, all_context, llm)
        if is_sufficient:
            break
        if attempt < max_attempts:
            current_query = reformulate_query(question, all_context, llm)

    answer = generate_answer(question, all_context, llm)
    elapsed = time.time() - start
    return {"question": question, "answer": answer, "retrievals": total_retrievals, "time_taken": round(elapsed, 2)}