FROM python:3.14.0-slim

# 设置时区和语言环境
ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

# 安装系统依赖和中文字体
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-wqy-zenhei \
    ttf-wqy-zenhei \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# 更新字体缓存
RUN fc-cache -fv

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]