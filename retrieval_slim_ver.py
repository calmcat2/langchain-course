from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
import os


load_dotenv()


def format_docs(docs):
    """Format the documents for the prompt."""
    return "\n\n".join([doc.page_content for doc in docs])


if __name__ == "__main__":
    print("Creating a llm...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    query = "what is RAG?"

    print("Embedding...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")

    print("Creating a vector store object...")
    vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX"], embedding=embeddings
    )

    print("Creating a RAG chain...")
    prompt_template = """
    Answer any use questions based solely on the context below:

    <context>
    {context}
    </context>
    Just say "Don't know" if you don't know the answer.
    Always say "thank you" at the end of your answer.
    """

    prompt = PromptTemplate.from_template(template=prompt_template)
    rag_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    print("Retrieving...")
    result = rag_chain.invoke(query)
    print(result)
