from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = FastAPI(
    title="T-Bank Logo Detection API",
    description="API for detecting T-Bank logo on images",
    version="1.0.0"
)


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "weights", "best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

model = YOLO(MODEL_PATH)
print("✅ Модель YOLO загружена успешно")


class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class Detection(BaseModel):
    bbox: BoundingBox


class DetectionResponse(BaseModel):
    detections: List[Detection]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


def detect_logo_on_image(image: np.ndarray) -> List[List[int]]:
    try:
        results = model(image, imgsz=640, conf=0.25, device='cpu')  # можно 'cuda', если GPU
        boxes = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes.xyxy.cpu().numpy():
                    x_min, y_min, x_max, y_max = map(int, box[:4])
                    boxes.append([x_min, y_min, x_max, y_max])

        return boxes

    except Exception as e:
        print(f"Ошибка при детекции: {e}")
        return []


@app.post("/detect", response_model=DetectionResponse)
async def detect_logo(file: UploadFile = File(...)):
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Поддерживаемые форматы: JPEG, PNG, BMP, WEBP. Получено: {file.content_type}"
        )

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

        bboxes = detect_logo_on_image(image)

        detections = [
            Detection(
                bbox=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3])
            )
            for box in bboxes
        ]

        return DetectionResponse(detections=detections)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")


@app.get("/")
def health():
    return {"status": "ok", "message": "API работает. Используй POST /detect"}