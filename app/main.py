from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os

from app.api.routes import router as api_router

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Industrial Safety System API")

# Cấu hình CORS để Frontend gọi được API mà không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nhúng API Router từ routes.py
app.include_router(api_router)

# Phục vụ các file tĩnh cho Frontend (HTML, CSS, JS)
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    # Khi người dùng truy cập trang chủ "/", tự động điều hướng tới giao diện web
    @app.get("/")
    def read_root():
        return RedirectResponse(url="/static/index.html")
else:
    @app.get("/")
    def read_root():
        return {"message": "Hệ thống AI đang hoạt động tốt. (Lưu ý: Không tìm thấy thư mục frontend)."}
