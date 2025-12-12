import os 
import time 
import threading 
import uuid
import asyncio
from typing import Dict, List, Any, Optional, AsyncIterator
from agent import Agent 
from config import settings 
from runtime_config import runtime_config 
from rag_manage import RagManager 
try:
    from workflow.correction_workflow import CorrectionWorkflow 
except ImportError:
    CorrectionWorkflow = None

class SessionManager: 
    """会话管理器 - 集成工作流（优化版）""" 

    def __init__(self): 
        self.agent = Agent()  # 智能体 
        self.rag_manager = RagManager(self.agent) # RAG管理器 
        self.workflow = None 
        self.active_workflows: Dict[str, CorrectionWorkflow] = {}  # 活跃工作流实例 
        self.file_tasks: Dict[str, Dict[str, Any]] = {} # 文件处理任务状态

        self.cleanup_interval = 3600 
        self.expiry_hours = 3 

        self._start_cleanup_thread() 
    
    async def chat_with_rag_stream(self, session_id: str, user_input: str, enable_net_search: bool = False, file_paths: Optional[List[str]] = None) -> AsyncIterator[str]:
        """
        RAG聊天 (流式) - 代理到RagManager
        """
        async for chunk in self.rag_manager.chat_with_rag_stream(session_id, user_input, enable_net_search, file_paths=file_paths):
            yield chunk

    def submit_file_task(self, session_id: str, file_path: str, skip_summary: bool = False) -> str:
        """提交文件处理任务"""
        task_id = str(uuid.uuid4())
        self.file_tasks[task_id] = {
            "status": "processing", 
            "start_time": time.time(),
            "session_id": session_id,
            "file_path": file_path
        }
        # 启动后台任务
        asyncio.create_task(self._process_file_task(task_id, session_id, file_path, skip_summary))
        return task_id

    async def _process_file_task(self, task_id: str, session_id: str, file_path: str, skip_summary: bool = False):
        """处理文件任务"""
        try:
            # 调用原本的添加文档逻辑
            # 注意：rag_manager.add_document 返回的是列表 [{"status":..., "message":...}]
            res = await self.add_document_to_session(session_id, file_path, skip_summary)
            
            # 检查结果
            success = False
            details = {}
            if res and isinstance(res, list) and len(res) > 0:
                first_res = res[0]
                if first_res.get("status") == "success":
                    success = True
                details = first_res
            
            self.file_tasks[task_id]["status"] = "completed" if success else "failed"
            self.file_tasks[task_id]["result"] = res
            self.file_tasks[task_id]["end_time"] = time.time()
            
        except Exception as e:
            self.file_tasks[task_id]["status"] = "failed"
            self.file_tasks[task_id]["error"] = str(e)
            self.file_tasks[task_id]["end_time"] = time.time()

    def get_file_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        return self.file_tasks.get(task_id, {"status": "not_found"})

    def _start_cleanup_thread(self): 
        """启动清理线程""" 
 
        def cleanup_worker(): 
            while True: 
                try: 
                    # 清理RAG管理器中的过期检索器 
                    sessions_to_clean = self.rag_manager.cleanup_expired_retrievers() 
 
                    for session_id in sessions_to_clean: 
                        self.cleanup_session(session_id) 
 
                    time.sleep(self.cleanup_interval) 
                except Exception: 
                    time.sleep(self.cleanup_interval) 
 
        thread = threading.Thread(target=cleanup_worker, daemon=True) 
        thread.start() 
 
    async def add_document_to_session(self, session_id: str, file_path: str, skip_summary: bool = False) -> List: 
        """ 
        添加文档到会话 
        """ 
        return await self.rag_manager.add_document(session_id, file_path, skip_summary) 
 
    async def get_or_create_retriever(self, session_id: str, file_paths: Optional[List[str]] = None, ephemeral: bool = False) -> Any: 
        """ 
        获取或创建检索器 (代理到RagManager) 
        """ 
        return await self.rag_manager.get_or_create_retriever(session_id, file_paths, ephemeral) 
 
    async def chat_with_rag(self, session_id: str, user_input: str, enable_net_search: bool = False, file_paths: Optional[List[str]] = None) -> str: 
        """ 
        RAG聊天 (代理到RagManager) 
        """ 
        return await self.rag_manager.chat_with_rag(session_id, user_input, enable_net_search, file_paths=file_paths) 
 
    async def chat_after_adding_file(self, session_id: str, user_input: str) -> str: 
        """ 
        添加文件后的聊天 
        """ 
        # 直接调用智能体，不再切换Prompt 
        return await self.agent.chat( 
            session_id=session_id, 
            user_input=user_input 
        ) 
 
    def cleanup_session(self, session_id: str): 
        """ 
        清理会话 
        """ 
        self.rag_manager.cleanup_session_retriever(session_id) 
        self.agent.clear_session(session_id) 
 
    def get_session_info(self, session_id: str) -> dict: 
        """ 
        获取会话信息 
        """ 
        info = self.rag_manager.get_retriever_info(session_id) 
        info["session_id"] = session_id 
        if info["has_retriever"]: 
            info["status"] = "active" 
        else: 
            info["status"] = "inactive" 
        return info 
 
    def list_available_features(self, session_id: str) -> list: 
        """
        列出当前会话可用的功能特性
        
        Args:
            session_id: 会话ID
            
        Returns:
            可用功能描述列表
        """
        features = [] 
        info = self.rag_manager.get_retriever_info(session_id) 
        has_docs = info.get('has_documents', False) 
        stats = info.get('retriever_stats', {}) 
        
        ocr_ok = bool(stats.get('ocr_available')) 
        vector_ok = bool(stats.get('vector_store_initialized') or stats.get('has_vector_store')) 
        
        if has_docs: 
            features.append("基于文档的检索与回答（RAG）") 
        else: 
            features.append("一般知识问答与教学建议") 
        if ocr_ok: 
            features.append("PDF/图片OCR解析与向量检索") 
        else: 
            features.append("PDF/图片解析（需配置OCR）") 
        if vector_ok: 
            features.append("语义向量检索与重排序") 
        features.append("TXT文档BM25检索") 
        features.append("混合检索器自动选择") 
        features.append("联网补充信息（文档不足时）") 
        features.append("会话历史查询与持久化") 
        features.append("会话清理与状态查看") 
        return features 

    def greet(self, session_id: str) -> str: 
        """
        生成欢迎语
        
        Args:
            session_id: 会话ID
            
        Returns:
            欢迎消息内容
        """
        self.rag_manager.ensure_session(session_id) 
        title = "你好呀！我是你的智能教学小助手 😊" 
        intro = "我可以陪你一起学习、备课，或者解答你遇到的各种学科问题。"
        
        features_list = [
            "📚 **解答学科问题**：数学、物理、化学、语文... 只要你问，我就能答！",
            "📝 **批改作业**：上传作业图片或文件，我帮你检查对错，还能分析解题思路。",
            "🧠 **讲解知识点**：哪里不会点哪里，我会用通俗易懂的语言为你讲解。",
            "🔎 **联网搜索**：最新的考试动态、教育资讯，我都能帮你查到。",
            "📂 **文档助手**：上传课件或资料，我可以帮你总结重点，还能基于文档回答问题。(支持pdf，txt，图片)"
        ]
        
        bullets = "\n".join([f"- {f}" for f in features_list]) 
        return f"{title}\n\n{intro}\n\n**我会做什么：**\n{bullets}\n\n随时告诉我你想做什么，或者直接把问题/文件发给我吧！🚀" 

    def get_session_history(self, session_id: str) -> list: 
        """
        获取会话历史记录
        
        Args:
            session_id: 会话ID
            
        Returns:
            历史消息列表
        """
        try: 
            return self.agent.get_session_history(session_id) 
        except Exception: 
            return [] 
 
    def list_all_sessions(self) -> Dict: 
        """ 
        列出所有活跃会话 
        """ 
        sessions = {} 
        # Get sessions from RagManager 
        # Since RagManager.retrievers_cache keys are the session IDs 
        for session_id in self.rag_manager.retrievers_cache.keys(): 
            sessions[session_id] = self.get_session_info(session_id) 
 
        return { 
            "total_sessions": len(sessions), 
            "active_sessions": len([s for s in sessions.values() if s.get('status') == 'active']), 
            "sessions": sessions 
        } 
 
    def get_session_stats(self) -> Dict: 
        """ 
        获取会话统计信息 
        """ 
        sessions = self.list_all_sessions() 
 
        total_documents = 0 
        for session in sessions['sessions'].values(): 
            if session.get('has_documents'): 
                total_documents += 1 
 
        return { 
            "total_sessions": sessions['total_sessions'], 
            "active_sessions": sessions['active_sessions'], 
            "sessions_with_documents": total_documents, 
            "cache_size": len(self.retrievers_cache) 
        } 
 
    def run_homework_workflow(self, session_id: str, file_list: List[str]) -> Dict[str, Any]: 
        """ 
        作业批改工作流入口 

        Args: 
            session_id: 会话ID 
            file_list: 待批改文件列表 

        Returns: 
            结果字典，包含批改结果数据
        """ 
        if CorrectionWorkflow is None:
            return {"error": "CorrectionWorkflow module not found", "session_id": session_id}
            
        try: 
            wf = CorrectionWorkflow(session_id=session_id) 
            self.active_workflows[session_id] = wf 
            result = wf.batch_correct(file_list) 
            # 完成后移除（或保留一段时间？这里先移除） 
            if session_id in self.active_workflows: 
                del self.active_workflows[session_id] 
            return result 
        except Exception as e: 
            if session_id in self.active_workflows: 
                del self.active_workflows[session_id] 
            return {"error": str(e), "session_id": session_id, "files": file_list} 
 
    def get_homework_progress(self, session_id: str) -> Dict[str, Any]: 
        """获取作业批改进度""" 
        if session_id in self.active_workflows: 
            return self.active_workflows[session_id].get_progress(session_id) 
        return {} 
