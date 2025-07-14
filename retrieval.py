from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os


load_dotenv()

if __name__ == "__main__":
    print("Creating a llm...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    query = "what is the best vector store for knowledge base?"

    print("Embedding...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")

    print("Creating a vector store object...")
    vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX"], embedding=embeddings
    )

    print("Retrieving...")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke(input={"input": query})
    print(result)
