from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dashboard.routes_api import router as api_router
from dashboard.routes_pages import router as pages_router

DASHBOARD_DIR = Path(__file__).parent

app = FastAPI(title="BTC Mecha Predictor")
app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR / "static")), name="static")

templates = Jinja2Templates(directory=str(DASHBOARD_DIR / "templates"))

app.include_router(api_router, prefix="/api")
app.include_router(pages_router)
