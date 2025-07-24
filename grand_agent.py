from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.tools import Tool
from code_agent import code_agent_run
from csv_agent import csv_agent_run

load_dotenv()


tools = [
    Tool(
        name="Python agent",
        func=code_agent_run,
        description="""useful when you need to transform natural language to python and execute the python code,
                          returning the results of the code execution
                          DOES NOT ACCEPT CODE AS INPUT""",
    ),
    Tool(
        name="csv agent",
        func=csv_agent_run,
        description="""useful when you need to answer question over STOCK_US_XNAS_NVDA.csv file,
                         takes an input the entire question and returns the answer after running pandas calculations""",
    ),
]

base_prompt = hub.pull("langchain-ai/react-agent-template")
grand_agent = create_react_agent(
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
    prompt=base_prompt.partial(instructions=''),
    tools=tools,
)
agent_executor = AgentExecutor(agent=grand_agent,tools=tools,verbose=True)
print(agent_executor.invoke({"input": "generate a QR code for a random udemy course?"}))
print(agent_executor.invoke({"input": "how many days are counted in the csv file?"}))
