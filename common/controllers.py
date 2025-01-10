import cv2
import pytesseract
from typing import Union

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
        upscaled_image = cls.upscale_image(image)

        # Preprocess the image
        processed_image = cls.preprocess_image(upscaled_image)

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
        _, thresholded = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresholded