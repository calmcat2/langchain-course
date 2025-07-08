import os
import requests
from dotenv import load_dotenv

load_dotenv()


def get_linkedin_profile(linkedin_url: str, mock: bool = False) -> str:
    """
    Fetches LinkedIn profile information for a given LinkedIn URL.
    Args:
        linkedin_url (str): The LinkedIn profile URL.
        mock (bool): If True, returns a mock response instead of making an API call.

    Returns:
        str: A brief biography or profile information of the person.
    """
    if mock:
        data = requests.get(
            "https://gist.githubusercontent.com/emarco177/859ec7d786b45d8e3e3f688c6c9139d8/raw/5eaf8e46dc29a98612c8fe0c774123a7a2ac4575/eden-marco-scrapin.json"
        ).json()
    else:
        api_key = os.getenv("SCRAPIN_API_KEY")
        if not api_key:
            raise ValueError("Scrapin API key not found in environment variables.")

        api_endpoint = "https://api.scrapin.io/v1/enrichment/profile"

        paramters = {"linkedInUrl": linkedin_url, "apikey": api_key}
        response = requests.get(api_endpoint, params=paramters)

        if response.status_code == 200:
            data = response.json()
        else:
            return f"Error fetching profile: {response.status_code} - {response.text}"

    filtered_data = {p: v for p, v in data.items() if v not in [None, "", "N/A", "[]"]}

    return filtered_data["person"]


if __name__ == "__main__":
    linkedin_url = "https://www.linkedin.com/in/mengxinghe/"
    profile_info = get_linkedin_profile(linkedin_url, mock=True)
    print(profile_info)
