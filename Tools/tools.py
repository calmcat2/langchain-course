from langchain_tavily import TavilySearch
import os


def search_tavily(query: str) -> str:
    """
    Searches Tavily for the given query and returns the top result.

    Args:
        query (str): The search query.

    Returns:
        str: The top search result from Tavily.
    """
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    tavily_search = TavilySearch()
    response = tavily_search.run("LinkedIn profile " + query)

    if response and len(response["results"]) > 0:
        return response["results"][0]
    else:
        return "No results found."
