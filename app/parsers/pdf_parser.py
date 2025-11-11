import fitz  # pymupdf
from typing import List

class PDFParser:
    @staticmethod
    def parse(file_path: str) -> str:
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"PDF解析失败: {str(e)}")