
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from Tools.tools import search_tavily
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub



def lookup(name: str) -> str:
    """
    Looks up a person's name and returns the LinkedIn URL.
    Args:
        name (str): The name of the person to look up.

    Returns:
        str: The LinkedIn profile URL of the person.
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
    )

    template = """ Given the name {name}, find the LinkedIn profile URL of this person. 
                Return the URL only """

    prompt = PromptTemplate(input_variables=["name"], template=template)
    tools = [
        Tool(
            name="Crawl Google for linkedin profile",
            func=search_tavily,
            description="""Use this tool when you need to find the LinkedIn profile URL of a person by their name.
            The input should be the person's name, and the output will be a LinkedIn profile URL. Validate the URL before returning it.
            If the URL found is not valid then find a valid URL.""",
        )
    ]

    react_promt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, prompt=react_promt, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    result = agent_executor.invoke(input={"input": prompt.format(name=name)})
    return result["output"]


if __name__ == "__main__":
    name = "Elon Musk"
    linkedin_url = lookup(name)
    print(f"LinkedIn URL for {name}: {linkedin_url}")
