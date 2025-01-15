import os
import cv2
from fastapi import APIRouter, UploadFile, File
import numpy as np

common_router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@common_router.post('/upload_image', tags=['Common'])
async def upload_image(image: UploadFile = File(...)):
    try:
        from .controllers import ImageReimbursementController
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


@common_router.post('/upload-document', tags=['Common'])
def upload_docs(document: UploadFile = File(...)):
    try:
        from .controllers import LlamaCloudParsingController
        file_name = document.filename.removesuffix('.pdf')
        # job_id, status = LlamaCloudParsingController.upload_file(document)
        job_id, status = '3f90b680-fff9-4b15-aec7-ab60d62ee91b', 'PENDING'
        print(f"FILE UPLOADED SUCCESSFULLY JOB ID: {job_id} WITH STATUS: {status}")
        status = LlamaCloudParsingController.check_status(job_id)
        if status == "SUCCESS":
            print(f"LLAMA PARSING STATUS: {status}")
            parsed_result = LlamaCloudParsingController.get_parsed_result(job_id)
            if parsed_result:
                text = parsed_result['markdown']
                markdown_file_path = os.path.join(BASE_DIR, 'uploads', f"{file_name}")
                with open(markdown_file_path, 'w') as md_file:
                    md_file.write(text)
                    return {"status": f"FILE {file_name} SAVED IN UPLOADS"}
            return {"status": "NO RESULTS FOUND"}

    except Exception as e:
        print(f"UPLOAD DOCS EXC: ", e)
        return None
