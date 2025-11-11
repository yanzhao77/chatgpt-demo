import markdown
from typing import List
import re

class MDParser:
    @staticmethod
    def parse(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                # 转换为HTML然后去除标签，只保留文本
                html = markdown.markdown(md_content)
                # 使用更可靠的HTML标签去除方法
                text = re.sub(r'<[^>]+>', '', html)
                # 处理一些常见的HTML实体
                text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                # 处理换行
                text = text.replace('\n\n', '\n').strip()
                return text
        except Exception as e:
            raise Exception(f"MD解析失败: {str(e)}")