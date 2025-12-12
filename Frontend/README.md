# 智能教师助手前端项目 (Django)

本项目是基于 Django + Bootstrap 5 构建的现代化前端界面，用于与 `agent_rag` 后端 (FastAPI) 进行交互。

## 功能特性

- **会话管理**：创建、加入、删除会话（本地存储会话列表）。
- **智能问答**：基于 RAG 的实时聊天界面，支持 Markdown 渲染（后端返回文本）。
- **文件管理**：上传 PDF/图片/TXT 等文件到会话，查看已上传文件。
- **作业批改**：批量选择文件运行作业批改工作流，并查看结果。
- **响应式设计**：适配桌面和移动端。

## 项目结构

```
django_frontend/
├── django_frontend/    # Django 配置
├── web_ui/             # 应用逻辑 (Views)
├── templates/          # HTML 模板
│   ├── base.html       # 基础布局
│   ├── index.html      # 首页 (会话列表)
│   └── session_detail.html # 会话详情 (聊天/文件/工作流)
├── static/             # 静态资源
│   ├── css/            # 样式表
│   └── js/             # JavaScript (API 交互逻辑)
└── manage.py           # Django 管理脚本
```

## 部署说明

### 1. 环境准备

确保已安装 Python 3.8+。

```bash
# 在 E:\jiaoshi_ai 目录下 (或使用 venv)
pip install django requests python-dotenv django-cors-headers
```

### 2. 启动后端 (FastAPI)

前端依赖 `agent_rag` 后端服务。

```bash
cd E:\jiaoshi_ai
# 确保已安装后端依赖
# 启动 FastAPI (默认端口 8000)
python agent_rag/app.py
```

*注意：已修改 `agent_rag/app.py` 添加 CORS 支持，允许跨域访问。*

### 3. 启动前端 (Django)

```bash
cd E:\jiaoshi_ai\django_frontend
python manage.py runserver 8001
```

### 4. 访问

打开浏览器访问 [http://localhost:8001](http://localhost:8001)

## 配置说明

- **后端地址配置**：
  若后端地址变更，请修改 `static/js/api.js` 中的 `API_BASE_URL` 常量。

- **跨域设置**：
  Django 端已配置 `django-cors-headers` (开发模式允许所有)。
  FastAPI 端已配置 `CORSMiddleware` 允许所有来源。

## 开发规范

-遵循 PEP8 编码规范。
- 前后端分离：Django 负责渲染页面框架，JS 负责数据交互。
