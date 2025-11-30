"""Image processing utilities for vision analysis."""

import base64
import logging
from pathlib import Path
from typing import Optional

# Default MIME type for images
IMAGE_MIME_TYPE = "image/jpeg"

logger = logging.getLogger(__name__)

class ImageUtils:
    """Handles image encoding and validation."""

    @staticmethod
    def _get_mime_type(image_path: str) -> str:
        """
        Determine the MIME type of an image based on file extension.

        Args:
            image_path: Path to the image file

        Returns:
            MIME type string (e.g., "image/jpeg", "image/png")
        """
        suffix = Path(image_path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        return mime_types.get(suffix, IMAGE_MIME_TYPE)

    @staticmethod
    def encode_to_base64(image_path: str) -> str:
        """
        Convert an image file to base64 encoding.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image URL in data URI format

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If file is not a valid image
        """
        path = Path(image_path)

        if not path.exists():
            logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if not ImageUtils.is_valid_image(path):
            logger.error(f"Invalid image file: {image_path}")
            raise ValueError(f"File is not a valid image: {image_path}")

        try:
            with open(path, "rb") as file:
                file_data = file.read()
                encoded_file = base64.b64encode(file_data).decode("utf-8")
                mime_type = ImageUtils._get_mime_type(image_path)
                base64_url = f"data:{mime_type};base64,{encoded_file}"
            return base64_url
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise

    @staticmethod
    def is_valid_image(path: Path) -> bool:
        """
        Check if a file is a valid image based on extension.

        Args:
            path: Path object to the file

        Returns:
            True if file has a valid image extension, False otherwise
        """
        valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        return path.suffix.lower() in valid_extensions
