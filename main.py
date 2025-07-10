import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Add the project root to sys.path to resolve module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Agents.linkedin_scraper import lookup
from Third_parties.linkedin import get_linkedin_profile
from dotenv import load_dotenv
from output_parser import summary_parser, Summary


def ice_breaker_start(name: str = "Eden Marco", mock: bool = True) -> tuple[Summary, str]:
    summary_template = """
    given the Linkedin information {info} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    \n {format_instructions}"""
    prompt = PromptTemplate(
        input_variables=["info"],
        template=summary_template,
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()
        },
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    chain = prompt | llm | summary_parser

    linkedin_url = lookup(name)
    linkedin_data = get_linkedin_profile(linkedin_url, mock=mock)
    response = chain.invoke(input={"info": linkedin_data})
    print(response)

    return response, linkedin_data["photoUrl"]


if __name__ == "__main__":
    load_dotenv()

    print("Starting the Ice Breaker...")
    ice_breaker_start(name="Eden Marco", mock=True)
