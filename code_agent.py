from dotenv import load_dotenv
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

load_dotenv()


def code_agent_run(query: str):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    tools = [PythonREPLTool()]

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    prompt_template = hub.pull("langchain-ai/react-agent-template")
    prompt = prompt_template.partial(instructions=instructions)

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_executor.invoke({"input": query})


if __name__ == "__main__":
    query = """generate 5 QRcode in the current diretory that directs to random Udemy courses. 
                    You have Qrcode package installed already."""
    code_agent(query)
