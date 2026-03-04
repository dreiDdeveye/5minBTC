from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@router.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/backtest")
def backtest_page(request: Request):
    return templates.TemplateResponse("backtest.html", {"request": request})


@router.get("/model")
def model_page(request: Request):
    return templates.TemplateResponse("model.html", {"request": request})
