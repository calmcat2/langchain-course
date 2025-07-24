from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_csv_agent


def csv_agent_run(query: str):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    agent = create_csv_agent(
        llm,
        "STOCK_US_XNAS_NVDA.csv",
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )

    result = agent.invoke({"input": query})
    return result["output"]


if __name__ == "__main__":
    res = csv_agent_run(
        "from the csv file, 1. find the highest price of the stock. 2. find the lowest price of the stock"
    )
    print(res)
