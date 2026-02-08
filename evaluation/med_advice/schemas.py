from pydantic import BaseModel, Field


class JudgeOutput(BaseModel):
    reasoning: str = Field(description="Explanation of the verdict")
    verdict: str = Field(description="PASS or FAIL")
