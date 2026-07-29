"""Server-rendered read-only evidence-review pages."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.config import Settings
from app.ui.config import PublicUiConfig, build_public_ui_config

TEMPLATE_ROOT = Path(__file__).with_name("templates")
templates = Jinja2Templates(directory=TEMPLATE_ROOT)
UI_UNAVAILABLE_HTML = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1">'
    "<title>AeroDocs Evidence Review unavailable</title></head><body>"
    "<main><h1>AeroDocs Evidence Review is unavailable</h1>"
    "<p>Core AeroDocs services remain available.</p></main></body></html>"
)


def create_ui_router(settings: Settings) -> APIRouter:
    """Create page routes containing no service or storage dependencies."""
    router = APIRouter(prefix="/ui", include_in_schema=False)
    public_config = build_public_ui_config(settings)

    def context(request: Request, page: str) -> dict[str, object]:
        return {
            "request": request,
            "page": page,
            "ui_title": settings.ui_title,
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "ui_config": PublicUiConfig.model_validate(public_config),
        }

    @router.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="home.html",
            context=context(request, "home"),
        )

    @router.get("/documents", response_class=HTMLResponse)
    async def documents(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="documents.html",
            context=context(request, "documents"),
        )

    @router.get("/search", response_class=HTMLResponse)
    async def search(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="search.html",
            context=context(request, "search"),
        )

    @router.get("/ask", response_class=HTMLResponse)
    async def ask(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="ask.html",
            context=context(request, "ask"),
        )

    @router.get("/{unknown_path:path}", response_class=HTMLResponse)
    async def ui_not_found(request: Request, unknown_path: str) -> HTMLResponse:
        del unknown_path
        return templates.TemplateResponse(
            request=request,
            name="not_found.html",
            context=context(request, "not_found"),
            status_code=404,
        )

    return router


def create_ui_unavailable_router() -> APIRouter:
    """Create a fixed safe fallback for a typed UI configuration failure."""
    router = APIRouter(prefix="/ui", include_in_schema=False)

    @router.get("/{path:path}", response_class=HTMLResponse)
    async def unavailable(path: str) -> HTMLResponse:
        del path
        return HTMLResponse(UI_UNAVAILABLE_HTML, status_code=503)

    return router
