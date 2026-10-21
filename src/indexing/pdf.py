from pypdf import PdfReader
import os
from dotenv import load_dotenv
load_dotenv()


def extract_text_from_pdf():
    text = ""
    with open(os.getenv("SOURCE_PDF"), 'rb') as file:
        reader = PdfReader(file)
        print(f"Number of pages: {reader.pages}")
        for page in reader.pages:
            text += page.extract_text()
    return text


print(extract_text_from_pdf())
