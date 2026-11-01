from pypdf import PdfReader
import os
from dotenv import load_dotenv
load_dotenv()


def extract_text_from_pdf():
    text = ""
    with open(os.getenv("SOURCE_PDF"), 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += (page.extract_text() or "")
    return text


if __name__ == "__main__":
    print(extract_text_from_pdf())
