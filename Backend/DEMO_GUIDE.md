# 后端演示与开发指南 (Backend Demo & Development Guide)

## 🛠️ 核心功能修复与优化 (Core Fixes & Optimizations)

### 修复与改进
1. **并发处理机制**
   - ✅ **异步任务队列**: 实现 `BackgroundTasks` 处理文件解析，上传接口实现毫秒级响应。
   - ✅ **参数优化**: 引入 `skip_summary=true` 参数，允许前端跳过耗时的摘要生成步骤，大幅提升批量上传体验。

2. **路径与存储管理**
   - ✅ **统一存储路径**: 所有用户数据（上传、结果、记忆）强制重定向至 `Backend/storage/`，解决路径混乱问题。
   - ✅ **绝对路径解析**: 使用 `os.path.abspath(__file__)` 修复不同目录启动导致的 `FileNotFoundError`。
   - ✅ **自动清理**: 实现了会话级别的资源清理接口。

3. **智能体逻辑增强**
   - ✅ **LangGraph 状态管理**: 引入状态图替代传统的 Chain，增强了多轮对话的稳定性。
   - ✅ **上下文限制**: 实现 30000 字符的历史记录滑动窗口裁剪，防止 LLM Token 溢出。
   - ✅ **隐私安全**: 敏感 API Key 抽离至 `.env`，并配置 `.gitignore` 防止泄露。

## 🎬 API 演示流程 (API Demo Flow)

### 完整的后端交互流程

#### 1. 会话初始化 (Session Init)
- **API**: `POST /sessions`
- **Payload**: `{"session_id": "可选自定义ID"}`
- **响应**: `{"session_id": "uuid", "message": "你好！我是你的AI助教..."}`
- **说明**: 初始化内存空间和 SQLite 索引。

#### 2. 文件上传与解析 (Upload & Parse)
- **API**: `POST /files/upload/async`
- **参数**: `file` (Multipart), `skip_summary=true`
- **流程**:
  1. **接收**: 文件流写入 `storage/uploads/{session_id}/`。
  2. **扫描**: 调用 Windows Defender 扫描病毒（安全保障）。
  3. **异步**: 立即返回 `task_id`，后台线程池启动 OCR 和向量化索引。
  4. **轮询**: 前端通过 `/files/status/{task_id}` 查询 `processing` -> `completed` 状态。

#### 3. 批量批改 (Batch Grading)
- **API**: `POST /grading/batch`
- **Payload**: `{"session_id": "...", "file_list": ["绝对路径1", "绝对路径2"]}`
- **流程**:
  - 启动 `GradingWorkflow` 工作流。
  - 并行处理每个作业文件。
  - **结果**: 为每个作业生成独立的 PDF 批改报告（已移除 CSV 汇总）。
  - **访问**: 报告可通过 `/grading_outputs/` 静态路径下载。

#### 4. 智能问答 (RAG Chat)
- **API**: `POST /sessions/{session_id}/chat`
- **Payload**: `{"question": "这个学生的弱项是什么？", "stream": true, "net_search": false}`
- **特性**:
  - **流式响应 (SSE)**: 实时输出 Token，实现“打字机”效果。
  - **混合检索**: 优先检索本地文档库，相关性不足时（如启用 `net_search`）自动调用 Tavily 联网搜索。
  - **引用溯源**: 返回精确的文档引用源（文件名、页码、文本片段）。

## ⚡ 技术架构特点 (Technical Architecture)

### RAG 引擎
- **LangChain + LangGraph**: 编排复杂的检索与生成流程，支持“循环”与“分支”逻辑。
- **混合检索 (Hybrid Search)**: 结合关键词搜索 (BM25) 与语义向量检索 (DashScope Embedding)。
- **多模态解析**: 集成 PaddleOCR，支持从图片和 PDF 中提取表格、公式和手写体文本。

### 性能设计
- **FastAPI 异步架构**: 使用 `async/await` 处理高并发 I/O 请求。
- **线程池隔离**: CPU 密集型计算（OCR/Embedding）在独立线程池中运行，不阻塞主事件循环。
- **环境隔离**: 通过 `venv` 和 `requirements.txt` 锁定 100+ 个依赖包的版本，确保可移植性。

## 🔧 真实数据流说明 (Real Data Flow)

与前端演示模式不同，后端处理均为**真实实时计算**：

1. **OCR**: 实时调用本地 PaddleOCR 模型或云端接口进行像素级识别。
2. **LLM**: 实时调用阿里云 Qwen-Turbo/Plus 模型生成回答。
3. **Search**: 实时调用 Tavily API 搜索最新互联网信息。

## 🚀 快速启动 (Quick Start)

```bash
# 1. 确保已配置 Backend/.env 文件 (参考 .env.example)

# 2. 启动服务
start-dev.bat

# 或者手动启动后端
cd Backend
.venv\Scripts\python -u app.py
```

## 📝 前端对接指南 (Frontend Integration)

前端开发者对接时需注意：

1. **上传接口对接**:
   - 务必设置 `skip_summary=true` 以获得极速上传体验。
   - 使用 `task_id` 轮询或 SSE 订阅 (`/files/subscribe/{task_id}`) 获取解析进度。

2. **聊天接口对接**:
   - 客户端需支持 `EventSource` 或流式读取。
   - 消息格式：`data: {"content": "..."}\n\n`。
   - 结束标记：`data: [END]\n\n`。

3. **文件路径**:
   - 后端返回的所有文件路径均为绝对路径，前端展示或下载时需通过后端的静态资源代理接口访问。

## ✅ 已完成的功能清单 (Completed Features)

- [x] **全链路路径修正**: 统一重定向到 Storage，修复所有 `FileNotFound` 问题。
- [x] **会话历史优化**: 自动裁剪 (30k chars) + SystemPrompt 保护。
- [x] **批改流程定制**: 移除 CSV 生成，专注于单文件 PDF 报告。
- [x] **安全增强**: Windows Defender 集成 + 敏感 Key 隔离。
- [x] **流式交互**: 标准 SSE 协议实现。
- [x] **部署友好**: 一键启动脚本 + 依赖版本锁定。

此文档旨在帮助开发者快速理解后端逻辑与交互方式，配合前端演示实现完整功能闭环。
