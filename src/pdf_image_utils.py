"""module for utilities for reading PDF and Images."""

from tqdm import tqdm
from pdf2image import convert_from_path

def convert_pdf_to_image(source_path: str, destination_path: str) -> None:
    """PDF to Image"""

    images = convert_from_path(source_path)
    for i in tqdm(range(len(images))):
        images[i].save(destination_path+"/"+'page'+ str(i) +'.jpg', 'JPEG')
