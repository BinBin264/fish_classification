from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
import os
from pathlib import Path
from app.inference import classify_fish
import numpy as np  # Import numpy để chuyển đổi ndarray
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from gradio_client import Client
from pydantic import BaseModel
import google.generativeai as genai

router = APIRouter()

@router.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Lưu tệp tin đã tải lên vào thư mục tạm thời
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        # Ghi tệp vào thư mục tạm thời
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Lớp phân loại cá từ hàm classify_fish
        label = classify_fish(file_path)

        # Nếu label là một mảng NumPy, chuyển nó thành list trước khi trả về
        if isinstance(label, np.ndarray):
            label = label.tolist()

        # Xóa tệp tin sau khi xử lý
        os.remove(file_path)

        # Trả về kết quả phân loại dưới dạng JSON response
        return JSONResponse(
            status_code=200,  # Mã trạng thái HTTP 200 cho yêu cầu thành công
            content={"label": label}  # Trả về dữ liệu phân loại dưới dạng JSON
        )

    except Exception as e:
        # Trả về thông báo lỗi nếu có ngoại lệ xảy ra
        print(f"Đã xảy ra lỗi: {str(e)}")
        return JSONResponse(
            status_code=500,  # Mã trạng thái HTTP 500 cho lỗi nội bộ
            content={"error": f"File processing failed: {str(e)}"}  # Trả về thông báo lỗi dưới dạng JSON
        )

# Thiết lập API key trực tiếp trong mã Python
os.environ["GEMINI_API_KEY"] = "AIzaSyAIVK3JFTRvDIjm6ePYjDivZsRo3nO0YTE"

# Cấu hình API key cho Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Tạo mô hình và cài đặt cấu hình
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# Định nghĩa lớp yêu cầu (Request) để nhận thông tin từ người dùng
class ChatRequest(BaseModel):
    message: str

# API để nhận tin nhắn và trả phản hồi từ Gemini
@router.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.message
    print(f"User message: {user_message}")

    # Bắt đầu một session chat với Gemini
    chat_session = model.start_chat(
        history=[]  # Bạn có thể thêm lịch sử trò chuyện nếu cần
    )
    
    # Gửi tin nhắn của người dùng và nhận phản hồi
    response = chat_session.send_message(user_message)

    # Trả kết quả về cho frontend
    return JSONResponse(content={"response": response.text})

