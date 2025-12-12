# Agentic RAG OCR Backend

多模态 RAG 系统后端服务，专为教师批量作业辅导教学设计，支持 PDF、图片和 TXT 文档的解析、索引、问答与智能批改。

## 技术栈

- **FastAPI**: 高性能 REST API 框架
- **PaddleOCR**: 多模态文档解析（文本、表格、图像、公式）
- **ChromaDB / FAISS**: 向量数据库与检索
- **LangChain & LangGraph**: RAG 编排与智能体状态管理
- **Alibaba Cloud DashScope (Qwen)**: 通义千问大语言模型
- **Tavily**: 联网搜索增强

## 功能特性

✅ **批量智能批改**: 支持并发上传 PDF/图片/TXT，自动生成逐份 PDF 批改报告
✅ **多模态解析**: 自动识别手写体、印刷体、表格与公式
✅ **智能问答 RAG**: 基于会话的文档问答，支持流式响应 (SSE)
✅ **会话管理**: 完整的会话历史记忆与上下文管理
✅ **异步处理**: 后台任务队列处理耗时操作，实时反馈进度

## 安装与启动

### 1. 环境准备

确保已安装 Python 3.10+。

```bash
cd Backend
# 创建虚拟环境
python -m venv .venv
# 激活虚拟环境 (Windows)
.venv\Scripts\activate
# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写配置：

```bash
# Windows
copy .env.example .env
```

编辑 `.env` 文件，填入必要的 API Key：

```env
# 阿里云百炼 API Key（必填 - 用于 LLM）
ALIYUNBAILIAN_API_KEY=your_key

# PaddleOCR Token（必填 - 用于 OCR）
PADDLEOCR_VL_TOKEN=your_token

# Tavily API Key（可选 - 用于联网搜索）
TAVILY_API_KEY=your_key

# LangSmith（可选 - 用于调试）
LANGSMITH_API_KEY=your_key
```

### 3. 启动服务

```bash
# 开发模式启动
python -u app.py
```

服务默认运行在 `http://0.0.0.0:8000`。
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

## API 接口概览

### 1. 会话管理
- **创建会话**: `POST /sessions`
- **获取历史**: `GET /sessions/{session_id}/history`
- **智能对话**: `POST /sessions/{session_id}/chat`
  - 支持流式输出 (`stream=true`)
  - 支持联网搜索 (`net_search=true`)

### 2. 文件上传
- **同步上传**: `POST /files/upload`
- **异步上传**: `POST /files/upload/async`
- **查询状态**: `GET /files/status/{task_id}`

### 3. 批量批改
- **启动批改**: `POST /grading/batch`
  ```json
  {
    "session_id": "session_uuid",
    "file_list": ["path/to/file1.pdf", "path/to/file2.jpg"]
  }
  ```
- **查询进度**: `GET /grading/status/{task_id}`

## 项目结构

```
Backend/
├── app.py                  # FastAPI 应用入口
├── agent.py                # LangChain 智能体与 RAG 逻辑
├── grading_workflow.py     # 批量批改工作流核心逻辑
├── office_processor.py     # PDF/Word/Image 处理工具
├── txt_processor.py        # 文本处理工具
├── config.py               # 全局配置
├── requirements.txt        # Python 依赖清单
├── .env.example            # 环境变量模版
└── storage/                # 数据持久化目录
    ├── uploads/            # 上传文件存储
    ├── output/             # 批改结果输出
    └── session_memory/     # 会话记忆存储
```

## 数据存储

- **上传文件**: 存储于 `Backend/storage/uploads/{session_id}/`
- **批改报告**: 存储于 `Backend/storage/output/{task_id}/`
- **会话记忆**: JSON 格式存储于 `Backend/storage/session_memory/`

## 常见问题 (FAQ)

**Q: 必要的 API Key 在哪里获取？**

*   **阿里云百炼 (LLM)**: [点击前往控制台](https://bailian.console.aliyun.com/?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.2.66357b08LUz5q3&tab=demohouse#/api-key)
    *   用途：提供通义千问大模型服务。
*   **Tavily (搜索)**: [点击前往 Dashboard](https://app.tavily.com/home)
    *   用途：提供 AI 优化的联网搜索能力。
*   **飞桨 (PaddleOCR)**: [点击获取 Access Token](https://aistudio.baidu.com/loginmid?redirectUri=https%3A%2F%2Faistudio.baidu.com%2Faccount%2FaccessToken)
    *   用途：用于高精度的文档 OCR 解析。
*   **LangSmith (调试)**: [点击前往官网](https://smith.langchain.com/)
    *   用途：(可选) 用于链路追踪和性能监控。

**Q: 启动时提示端口被占用？**
A: 请检查是否有其他服务占用了 8000 端口，或者尝试修改 `app.py` 中的启动端口（如果使用 uvicorn 命令行启动）。

**Q: 上传文件大小有限制吗？**
A: 默认限制为 100MB。可以在 `app.py` 中修改 `MAX_UPLOAD_SIZE` 常量。

## License

MIT
