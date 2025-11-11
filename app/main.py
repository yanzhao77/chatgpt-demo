from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
import tempfile
import uuid
from zhipuai import ZhipuAI
from typing import Dict, List, Optional
import asyncio
import time
import re

# 导入自定义模块
from app.models.embedding import EmbeddingModel
from app.rag.core import RAGCore
from app.parsers.pdf_parser import PDFParser
from app.parsers.docx_parser import DOCXParser
from app.parsers.txt_parser import TXTParser
from app.parsers.md_parser import MDParser
load_dotenv()

app = FastAPI(title="文档问答系统")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 设置模板目录
templates = Jinja2Templates(directory="app/templates")

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

# 存储会话和最后访问时间
sessions: Dict[str, RAGCore] = {}
session_last_access: Dict[str, float] = {}

# 会话超时时间（秒），例如2小时
SESSION_TIMEOUT = 2 * 60 * 60

def cleanup_expired_sessions():
    """清理过期会话"""
    current_time = time.time()
    expired_sessions = [
        session_id for session_id, last_access_time in session_last_access.items()
        if current_time - last_access_time > SESSION_TIMEOUT
    ]
    
    for session_id in expired_sessions:
        sessions.pop(session_id, None)
        session_last_access.pop(session_id, None)
    
    if expired_sessions:
        print(f"清理了 {len(expired_sessions)} 个过期会话")

# 定期清理过期会话的任务
async def periodic_cleanup():
    while True:
        cleanup_expired_sessions()
        await asyncio.sleep(600)  # 每10分钟检查一次

# 初始化embedding模型
embedding_model = EmbeddingModel()

# 在应用启动时启动清理任务
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_cleanup())

# 从环境变量获取API Key
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
if not ZHIPU_API_KEY:
    raise ValueError("请设置 ZHIPU_API_KEY 环境变量")

# 初始化ZhipuAI客户端
zhipu_client = ZhipuAI(api_key=ZHIPU_API_KEY)

# 定义使用的模型
ZHIPUAI_MODEL = "glm-4-flash"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def chunk_text_semantically(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    根据语义分块文本，尽量在句子边界处分割
    
    Args:
        text: 要分块的文本
        chunk_size: 每块的目标大小
        overlap: 重叠大小
        
    Returns:
        分块后的文本列表
    """
    import re
    
    # 按句子分割（以句号、感叹号、问号等为界）
    sentences = re.split(r'(?<=[。！？\n])', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 如果添加当前句子会使块超过大小，则保存当前块并开始新块
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            # 为下一区块保留重叠部分
            current_chunk = current_chunk[-overlap:] + sentence
        else:
            current_chunk += sentence
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # 检查文件大小
    if file.size > 20 * 1024 * 1024:  # 20MB
        raise HTTPException(status_code=400, detail="文件大小不能超过20MB")
    
    # 检查文件类型
    allowed_types = ["text/plain", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    if file.content_type not in allowed_types and not file.filename.endswith('.md'):
        raise HTTPException(status_code=400, detail="不支持的文件类型")
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        temp_file_path = tmp_file.name
    
    try:
        # 解析文件
        text = ""
        if file.filename.endswith('.pdf'):
            text = PDFParser.parse(temp_file_path)
        elif file.filename.endswith('.docx'):
            text = DOCXParser.parse(temp_file_path)
        elif file.filename.endswith('.txt'):
            text = TXTParser.parse(temp_file_path)
        elif file.filename.endswith('.md'):
            text = MDParser.parse(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="不支持的文件类型")
        
        # 创建新的会话ID
        session_id = str(uuid.uuid4())
        
        # 创建RAG核心实例
        rag_core = RAGCore()
        rag_core.set_embedding_model(embedding_model)
        
        # 文本分块（按语义分块）
        chunks = chunk_text_semantically(text, chunk_size=800, overlap=100)
        
        # 添加到索引
        rag_core.add_texts(chunks)
        
        # 存储会话
        sessions[session_id] = rag_core
        session_last_access[session_id] = time.time()
        
        return {"session_id": session_id, "chunk_count": len(chunks)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)


@app.post("/chat/")
async def chat(request: ChatRequest):
    question = request.question
    session_id = request.session_id
    # 如果提供了session_id，则使用RAG流程
    if session_id:
        # 检查会话是否存在
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # 更新会话最后访问时间
        session_last_access[session_id] = time.time()
        
        rag_core = sessions[session_id]
        
        # 检索相关文本，先检索9个候选文本块用于rerank
        results = rag_core.search(question, k=9)
        
        # 构建提示词
        context = "\n".join([f"相关文本 {i+1}: {text}" for i, (text, score) in enumerate(results[:3])])
        prompt = f"你是一个智能助手，请根据以下上下文回答问题。如果无法从上下文中找到答案，请说\"抱歉，我无法根据提供的信息回答这个问题。\"\n\n上下文：\n{context}\n\n问题：{question}\n\n回答："
    else:
        # 如果没有提供session_id，则直接与模型对话
        prompt = f"你是一个智能助手，请回答以下问题：\n\n问题：{question}\n\n回答："
    
    try:
        # 调用GLM-4-Flash API
        response = zhipu_client.chat.completions.create(
            model=ZHIPUAI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {"answer": response.choices[0].message.content}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调用大模型失败: {str(e)}")
