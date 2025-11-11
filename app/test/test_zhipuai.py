import os
import sys
import asyncio

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 在加载dotenv之前设置HF_ENDPOINT
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
from zhipuai import ZhipuAI

# 加载环境变量
load_dotenv()

# 定义使用的模型
ZHIPUAI_MODEL = "glm-4-flash"

def test_zhipu_api_connection():
    """
    测试与智谱AI API的连接
    """
    print("开始测试智谱AI API连接...")
    
    # 从环境变量获取API Key
    zhipu_api_key = os.getenv("ZHIPU_API_KEY")
    
    if not zhipu_api_key:
        print("错误: 未找到ZHIPU_API_KEY环境变量")
        print("请确保已设置API密钥")
        return False
    
    try:
        # 初始化ZhipuAI客户端
        client = ZhipuAI(api_key=zhipu_api_key)
        print("✓ ZhipuAI客户端初始化成功")
        
        # 发送测试请求
        print("发送测试请求...")
        response = client.chat.completions.create(
            model=ZHIPUAI_MODEL,
            messages=[
                {"role": "user", "content": "你好，请简单介绍一下 yourself"}
            ],
            max_tokens=100
        )
        
        # 检查响应
        if response and response.choices and len(response.choices) > 0:
            print("✓ API请求成功")
            print(f"模型响应: {response.choices[0].message.content}")
            print("✓ 智谱AI API连接测试通过")
            return True
        else:
            print("✗ API请求失败: 响应为空或格式不正确")
            return False
            
    except Exception as e:
        if "429" in str(e):
            print("✗ API请求失败: 余额不足或无可用资源包，请充值")
        else:
            print(f"✗ API请求失败: {str(e)}")
        return False

def test_async_zhipu_api_connection():
    """
    测试与智谱AI API的连接（异步方式）
    注意：zhipuai SDK本身不支持异步，这里只是模拟异步环境下的调用
    """
    print("开始异步环境测试智谱AI API连接...")
    
    # 从环境变量获取API Key
    zhipu_api_key = os.getenv("ZHIPU_API_KEY")
    
    if not zhipu_api_key:
        print("错误: 未找到ZHIPU_API_KEY环境变量")
        print("请确保已设置API密钥")
        return False
    
    try:
        # 初始化ZhipuAI客户端
        client = ZhipuAI(api_key=zhipu_api_key)
        print("✓ ZhipuAI客户端初始化成功")
        
        # 发送测试请求
        print("发送异步环境测试请求...")
        response = client.chat.completions.create(
            model=ZHIPUAI_MODEL,
            messages=[
                {"role": "user", "content": "请用一句话描述人工智能的未来"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        # 检查响应
        if response and hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                print("✓ 异步环境API请求成功")
                print(f"模型响应: {choice.message.content}")
                print("✓ 智谱AI API异步环境测试通过")
                return True
            else:
                print("✗ 异步环境API请求失败: 响应消息格式不正确")
                return False
        else:
            print("✗ 异步环境API请求失败: 响应为空或格式不正确")
            return False
            
    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg:
            print("✗ 异步环境API请求失败: 余额不足或无可用资源包，请充值")
        elif "api key" in error_msg or "unauthorized" in error_msg:
            print("✗ 异步环境API请求失败: API密钥无效或未授权")
        elif "network" in error_msg or "connection" in error_msg:
            print("✗ 异步环境API请求失败: 网络连接问题")
        else:
            print(f"✗ 异步环境API请求失败: {str(e)}")
        return False

def test_embedding_model():
    """
    测试嵌入模型功能
    """
    print("开始测试嵌入模型...")
    
    try:
        from app.models.embedding import EmbeddingModel
        import numpy as np
        
        # 初始化嵌入模型
        embedding_model = EmbeddingModel()
        print("✓ 嵌入模型初始化成功")
        
        # 测试编码功能
        test_texts = [
            "这是一个测试句子。",
            "这是另一个测试句子。"
        ]
        
        embeddings = embedding_model.encode(test_texts)
        print(f"✓ 文本编码成功，生成了形状为 {embeddings.shape} 的嵌入向量")
        
        # 检查向量维度
        if isinstance(embeddings, np.ndarray) and len(embeddings.shape) == 2:
            print("✓ 嵌入向量格式正确")
            return True
        else:
            print("✗ 嵌入向量格式不正确")
            return False
            
    except Exception as e:
        print(f"✗ 嵌入模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_with_zhipu():
    """
    测试RAG流程与智谱AI API的集成
    """
    print("开始测试RAG与智谱AI API集成...")
    
    try:
        # 导入必要的模块
        from app.rag.core import RAGCore
        from app.models.embedding import EmbeddingModel
        
        # 初始化嵌入模型
        embedding_model = EmbeddingModel()
        print("✓ 嵌入模型初始化成功")
        
        # 创建RAG核心实例
        rag_core = RAGCore()
        rag_core.set_embedding_model(embedding_model)
        print("✓ RAG核心初始化成功")
        
        # 添加测试文本
        test_texts = [
            "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
            "机器学习是人工智能的一个子集，它使计算机能够从数据中学习并做出决策或预测。",
            "深度学习是机器学习的一个分支，它模仿人脑的工作方式处理数据和创建模式，用于决策制定。"
        ]
        rag_core.add_texts(test_texts)
        print("✓ 测试文本添加成功")
        
        # 测试检索功能
        query = "什么是人工智能？"
        results = rag_core.search(query, k=3)
        print("✓ 文本检索功能正常")
        
        # 构建提示词
        context = "\n".join([f"相关文本 {i+1}: {text}" for i, (text, score) in enumerate(results)])
        prompt = f"你是一个智能助手，请根据以下上下文回答问题。如果无法从上下文中找到答案，请说\"抱歉，我无法根据提供的信息回答这个问题。\"\n\n上下文：\n{context}\n\n问题：{query}\n\n回答："
        print("✓ 提示词构建成功")
        
        # 从环境变量获取API Key
        zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        if not zhipu_api_key:
            print("错误: 未找到ZHIPU_API_KEY环境变量")
            return False
        
        # 初始化ZhipuAI客户端
        client = ZhipuAI(api_key=zhipu_api_key)
        
        # 发送请求到大模型
        response = client.chat.completions.create(
            model=ZHIPUAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        
        if response and hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                print("✓ RAG与大模型集成测试成功")
                print(f"模型响应: {choice.message.content}")
                return True
            else:
                print("✗ RAG与大模型集成测试失败: 响应消息格式不正确")
                return False
        else:
            print("✗ RAG与大模型集成测试失败: 响应为空或格式不正确")
            return False
            
    except Exception as e:
        if "429" in str(e):
            print("✗ RAG与大模型集成测试失败: 余额不足或无可用资源包，请充值")
        else:
            print(f"✗ RAG与大模型集成测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("大模型连接测试")
    print("=" * 50)
    
    # 测试基本连接
    basic_test_passed = test_zhipu_api_connection()
    print()
    
    # 测试异步环境连接
    async_test_passed = test_async_zhipu_api_connection()
    print()
    
    # 测试嵌入模型
    embedding_test_passed = test_embedding_model()
    print()
    
    # 测试RAG集成
    rag_test_passed = test_rag_with_zhipu()
    print()
    
    # 汇总结果
    print("=" * 50)
    print("测试结果汇总:")
    print(f"基本连接测试: {'通过' if basic_test_passed else '失败'}")
    print(f"异步环境测试: {'通过' if async_test_passed else '失败'}")
    print(f"嵌入模型测试: {'通过' if embedding_test_passed else '失败'}")
    print(f"RAG集成测试: {'通过' if rag_test_passed else '失败'}")
    
    all_tests_passed = basic_test_passed and async_test_passed and embedding_test_passed and rag_test_passed
    
    if all_tests_passed:
        print("所有测试通过! 大模型连接正常。")
    else:
        print("部分测试失败，请检查配置和网络连接。")
        
        # 提供故障排除建议
        if not (basic_test_passed and async_test_passed):
            print("\n故障排除建议:")
            print("1. 检查ZHIPU_API_KEY环境变量是否正确设置")
            print("2. 检查网络连接是否正常")
            print("3. 确认API密钥是否有效")
            print("4. 检查账户余额是否充足")
            
        if not (embedding_test_passed and rag_test_passed):
            print("\n故障排除建议:")
            print("1. 检查模型下载是否正常")
            print("2. 检查网络连接和HF_ENDPOINT设置")
            print("3. 确认模型缓存目录权限")
            
    print("=" * 50)