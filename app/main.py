from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os

from app.api.routes import router as api_router
from app.api.camera_sessions import router as camera_sessions_router
from app.services.mdns_service import mdns_service


@asynccontextmanager
async def lifespan(_app):
    await asyncio.to_thread(mdns_service.start)
    try:
        yield
    finally:
        await asyncio.to_thread(mdns_service.stop)

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Industrial Safety System API", lifespan=lifespan)

# Cấu hình CORS để Frontend gọi được API mà không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_static_asset_cache(request, call_next):
    """Prevent stale frontend assets while the dashboard is under development."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Nhúng API Router từ routes.py
app.include_router(api_router)
app.include_router(camera_sessions_router)

# Phục vụ các file tĩnh cho Frontend (HTML, CSS, JS)
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    # Khi người dùng truy cập trang chủ "/", tự động điều hướng tới giao diện web
    @app.get("/")
    def read_root():
        return RedirectResponse(url="/static/index.html?v=20261031-1")
else:
    @app.get("/")
    def read_root():
        return {"message": "Hệ thống AI đang hoạt động tốt. (Lưu ý: Không tìm thấy thư mục frontend)."}
