import cv2
import pytesseract
from typing import Union
import requests
from fastapi import UploadFile
from main import LLAMA_CLOUD_API_KEY, LLAMA_UPLOAD_URL, LLAMA_JOB_STATUS_URL, LLAMA_RESULT_URL

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class ImageReimbursementController:

    @classmethod
    def extract_text_from_image(cls, image):
        """
               Extracts text from the image using Tesseract OCR.

               Args:
                   image_path (str): Path to the image file.

               Returns:
                   str: Extracted text from the image.
        """
        # Upscale the image
        enhanced_image = cls.upscale_image(image)

        # Preprocess the image
        processed_image = cls.preprocess_image(enhanced_image)

        custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789'
        extracted_text = pytesseract.image_to_string(processed_image, config=custom_config)
        # Use Tesseract to extract text
        extracted_text = extracted_text.replace('—', '-')  # Fix long dashes
        extracted_text = extracted_text.replace('  ', ' ')

        return extracted_text

    @staticmethod
    def extract_amount_from_text(text: str) -> Union[float, None]:
        """
        Extracts an amount (number) from a string of text.

        Args:
            text (str): Extracted text from OCR.

        Returns:
            float: Extracted amount or None if not found.
        """
        # Look for any numerical values in the text
        import re

        # Regex pattern to find monetary values
        amount_pattern = r"\b\d+\.\d{2}\b"  # Matches numbers like 100.00
        amounts = re.findall(amount_pattern, text)

        if amounts:
            return float(amounts[-1])  # Return the first match as the amount

        return None

    @classmethod
    def upscale_image(cls, image, scale_factor=1.5):
        """
        Upscales the image to improve OCR accuracy.
        """
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize the image
        return cv2.resize(image, (new_width, new_height))

    @classmethod
    def preprocess_image(cls, image):
        """
        Preprocess the image to improve OCR accuracy.
        """
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Denoise using Gaussian Blur
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        denoised_image = cv2.fastNlMeansDenoising(gray_image)

        # Use thresholding to get binary image (black and white)
        _, threshold = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return threshold


class LlamaCloudParsingController:

    """ LLAMA CLOUD PARSING CONTROLLER FOR PARSING OF DOCUMENTS MD FILE """

    @classmethod
    def upload_file(cls, document: UploadFile):
        headers = {
            'Authorization': f'Bearer {LLAMA_CLOUD_API_KEY}',
            'accept': 'application/json'
        }

        file_content = document.file
        response = requests.post(LLAMA_UPLOAD_URL, headers=headers, files={'file': file_content})
        if response.status_code == 200:
            job_id = response.json().get('id')
            status = response.json().get('status')
            return job_id, status
        else:
            print("Failed to upload file:", response.json())
            return None

    @classmethod
    def check_status(cls, job_id):
        headers = {
            'Authorization': f'Bearer {LLAMA_CLOUD_API_KEY}',
            'accept': 'application/json'
        }

        response = requests.get(LLAMA_JOB_STATUS_URL.format(job_id=job_id), headers=headers)
        return response.json().get('status')

    @classmethod
    def get_parsed_result(cls, job_id):
        headers = {
            'Authorization': f'Bearer {LLAMA_CLOUD_API_KEY}',
            'accept': 'application/json'
        }

        response = requests.get(LLAMA_RESULT_URL.format(job_id=job_id), headers=headers)
        return response.json()

