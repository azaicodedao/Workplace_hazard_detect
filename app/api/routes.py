import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.inference.predictor import VideoPredictor
from app.schemas.prediction import PredictionResponse
import traceback

router = APIRouter()

# Khởi tạo mô hình AI toàn cục (Chạy 1 lần duy nhất lúc bật server)
try:
    predictor = VideoPredictor()
except Exception as e:
    print(f"Lỗi khởi tạo mô hình AI: {e}")
    predictor = None

# Thư mục lưu trữ video tạm thời
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/predict/", response_model=PredictionResponse)
async def predict_video(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Mô hình AI chưa được nạp thành công.")
        
    # Kểm tra định dạng file
    allowed_extensions = [".mp4", ".avi", ".mov"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Chỉ hỗ trợ các định dạng video: {allowed_extensions}")

    # Tạo đường dẫn lưu file tạm
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Lưu file từ request vào đĩa cứng
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể lưu video: {e}")
        
    # Gọi bộ não AI để dự đoán
    try:
        result = predictor.predict(file_path)
    except Exception as e:
        # Xóa file nếu xảy ra lỗi trong lúc phân tích
        if os.path.exists(file_path):
            os.remove(file_path)
        print("========== INFERENCE ERROR ==========")
        traceback.print_exc()
        print("=====================================")

        raise HTTPException(
            status_code=500,
            detail=f"Lỗi trong quá trình phân tích video: {str(e)}"
        )
        
    # Xóa file video sau khi phân tích xong để tiết kiệm dung lượng bộ nhớ
    if os.path.exists(file_path):
        os.remove(file_path)
        
    return result
