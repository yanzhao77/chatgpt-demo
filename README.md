# 📄 文档问答Web应用

> 一个基于FastAPI的智能文档问答系统，支持上传多种格式文档并进行自然语言问答

[![Python](https://img.shields.io/badge/Python-3.14.0+-3776AB?logo=python&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?logo=fastapi)]()
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)]()

![应用界面](/static/image/main.png)

## ✨ 功能特点

- 📁 支持上传 `.txt` / `.pdf` / `.docx` / `.md` 文件（≤20MB）
- 🔍 基于BGE中文embedding模型的精准语义检索
- 🧠 使用FAISS构建高效向量数据库
- 💬 调用GLM-4大模型生成自然流畅的回答
- 🌓 完美支持暗色模式与亮色模式切换
- 🖱️ 直观的拖拽式文件上传体验
- 📱 响应式设计，适配桌面与移动设备

## 🛠 技术栈

| 类别 | 技术 |
|------|------|
| **后端** | Python 3.14 + FastAPI |
| **前端** | HTMX + Tailwind CSS + Alpine.js |
| **向量模型** | BAAI/bge-small-zh-v1.5 |
| **向量库** | FAISS (CPU版) |
| **大模型** | 智谱AI glm-4-flash |
| **部署** | Docker + python:3.14.0-slim |

## 🚀 快速开始

### 前置条件
- Python 3.14.0+
- 智谱AI API Key ([注册获取](https://open.bigmodel.cn/))
- **网络要求**：由于需要从Hugging Face下载模型，建议确保网络畅通。如遇连接问题，系统已配置使用国内镜像站点加速下载。

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/yourusername/chatgpt-demo.git
cd chatgpt-demo

# 创建虚拟环境（推荐）
python -m venv venv
venv\\Scripts\\activate  # Windows
source venv/bin/activate  # Linux/MacOS

# 安装依赖
pip install -r requirements.txt

# 配置API密钥（两种方式任选其一）
# 方式1：创建 .env 文件
cp .env.example .env
# 然后编辑 .env 文件，填入你的API密钥

# 方式2：直接设置环境变量
export ZHIPU_API_KEY=your_api_key_here  # Linux/MacOS
set ZHIPU_API_KEY=your_api_key_here      # Windows

# 运行应用
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000` 查看应用界面

### Docker 部署

```bash
# 构建镜像
docker build -t chatgpt-demo .

# 运行容器
docker run -d -p 8000:8000 -e ZHIPU_API_KEY=your_key chatgpt-demo
```

## 📂 项目结构

```
/app
├── main.py                # FastAPI入口
├── models/                # 模型加载模块
│   └── embedding.py       # BGE模型单例实现
├── rag/                   # RAG核心逻辑
│   └── core.py            # 向量检索与reranker
├── parsers/               # 文件解析器
│   ├── pdf_parser.py      # pymupdf实现
│   ├── docx_parser.py
│   ├── txt_parser.py
│   └── md_parser.py
├── static/                # 静态资源
└── templates/             # HTML模板
    └── index.html         # 主界面（含暗色模式）
```

## 🤝 贡献指南

欢迎提交Issue和PR！请先阅读[贡献指南](CONTRIBUTING.md)。

## 📄 许可证

本项目基于 [MIT 许可证](LICENSE)。