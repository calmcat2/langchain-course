from typing import List, Dict, Any

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Summary(BaseModel):
    """
    A model to represent a summary and interesting facts about a person.
    """

    summary: str = Field(
        description="A short summary of the person's LinkedIn profile."
    )
    interesting_facts: List[str] = Field(
        description="Two interesting facts about the person."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary, "interesting_facts": self.interesting_facts}


summary_parser = PydanticOutputParser(pydantic_object=Summary)
