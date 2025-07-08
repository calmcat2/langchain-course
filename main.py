from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from Agents.linkedin_scraper import lookup
from Third_parties.linkedin import get_linkedin_profile
from dotenv import load_dotenv


def main(name: str = "Eden Marco", mock: bool = True):
    summary_template = """
    given the Linkedin information {info} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them"""
    prompt = PromptTemplate(input_variables=["info"], template=summary_template)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    chain = prompt | llm

    linkedin_url = lookup(name)
    linkedin_data = get_linkedin_profile(linkedin_url, mock=mock)
    response = chain.invoke(input={"info": linkedin_data})
    print(response)


if __name__ == "__main__":
    load_dotenv()

    print("Starting the Ice Breaker...")
    main(name="Eden Marco", mock=False)
