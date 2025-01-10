import cv2
import numpy as np
from fastapi import APIRouter, Request, UploadFile, File
from .controllers import ImageReimbursementController

common_router = APIRouter()


@common_router.post('/upload_image', tags=['Common'])
async def upload_image(image: UploadFile = File(...)):
    try:
        """
            Upload image, extract text from it using OCR.
        """
        image_data = await image.read()
        image_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Failed to decode the image"}

        # Extract text from the image using your controller
        extracted_text = ImageReimbursementController.extract_text_from_image(img)
        amount = ImageReimbursementController.extract_amount_from_text(extracted_text)

        return {"amount": amount}
    except Exception as e:
        print(f"UPLOAD IMAGE EXC: {e}")
        return None
