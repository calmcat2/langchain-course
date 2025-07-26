from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl
from langchain_core.documents import Document
from langchain_chroma import Chroma
import asyncio
from typing import List, Union

load_dotenv()


async def abatch_upload(
    split_docs: List[Document], vectorstore: Union[PineconeVectorStore, Chroma], batch_size: int = 20
):
    print("✅ Uploading content to vectorstore in batches...")
    total_batches = [
        split_docs[i : batch_size + i] for i in range(0, len(split_docs), batch_size)
    ]
    print(f"   Content is splitted into {len(total_batches)} batches.")

    async def aadd_batch(batch, batch_num):
        try:
            await vectorstore.aadd_documents(batch)
            print(f"- Successfully uploaded batch {batch_num}/{len(total_batches)}...")
        except Exception as e:
            print(f"❌Failed to upload batch {batch_num} - {e}")

    tasks = [aadd_batch(batch, i + 1) for i, batch in enumerate(total_batches)]
    await asyncio.gather(*tasks, return_exceptions=True)
    print("✅ All batches processed.")


async def main():
    print("✅ Initialization...")
    crawl_tool = TavilyCrawl(allow_external=False)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
    # vectorstore = PineconeVectorStore(
    #     embedding=embeddings, index_name=os.environ["PINECONE_INDEX"]
    # )
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    print("✅ Crawling in https://docs.langchain.com/")
    instructions = "Find documents related to AI Agents."
    try:
        crawl_result = crawl_tool.invoke(
            {
                "url": "https://docs.langchain.com/",
                "max_depth": 2,
                "extract_depth": "advanced",
                "include_images": False,
                "categories": ["Documentation"],
                "instructions": instructions,
            }
        )
    except Exception as e:
        print(f"❌ An error occured with crawler: {e}")
        return

    if isinstance(crawl_result, str) or crawl_result.get("results", []) == []:
        print(f"⚠️  No results are found for '{instructions}'. Existing...")
        return
    all_docs = crawl_result["results"]

    print("✅ Generating Documents based on the search result. ")
    documents = [
        Document(page_content=res["raw_content"], metadata={"source": res["url"]})
        for res in all_docs
    ]

    print("✅ Splitting documents...")
    split_docs = text_splitter.split_documents(documents)
    print(f"   Created {len(split_docs)} chunks")

    await abatch_upload(split_docs, vectorstore, 20)


if __name__ == "__main__":
    asyncio.run(main())
