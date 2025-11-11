from docx import Document
from typing import List

class DOCXParser:
    @staticmethod
    def parse(file_path: str) -> str:
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"DOCX解析失败: {str(e)}")