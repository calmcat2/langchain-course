from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

load_dotenv()

if __name__ == "__main__":
    file_path = "CKA_Curriculum_v1.33.pdf"
    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()   

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    texts = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("faiss_index_cka")

    read_vectorestore = FAISS.load_local(
        "faiss_index_cka", embeddings, allow_dangerous_deserialization=True
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0)
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        retriever=read_vectorestore.as_retriever(), combine_docs_chain=combine_docs_chain
    )   

    result = retrieval_chain.invoke({"input": "What is the CKA exam?"})
    print(result["answer"])