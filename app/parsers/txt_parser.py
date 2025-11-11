from typing import List

class TXTParser:
    @staticmethod
    def parse(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"TXT解析失败: {str(e)}")