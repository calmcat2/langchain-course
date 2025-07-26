import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("blog.txt")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(document)
    print(f"create {len(text)} chunks")

    print("creating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
    pinecone = PineconeVectorStore.from_documents(
        text, embeddings, index_name=os.environ["PINECONE_INDEX"]
    )
    print("finish")
