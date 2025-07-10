from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from main import ice_breaker_start

app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")


class IceBreakerRequest(BaseModel):
    name: str
    mock: bool


class IceBreakerResponse(BaseModel):
    summary: str
    interesting_facts: list[str]
    picture_url: str


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.post("/ice_breaker", response_model=IceBreakerResponse)
async def ice_breaker(request: IceBreakerRequest):
    summary_data, picture_url = ice_breaker_start(name=request.name, mock=request.mock)
    return IceBreakerResponse(
        summary=summary_data.summary,
        interesting_facts=summary_data.interesting_facts,
        picture_url=picture_url
    )
