import os
import time
import json
import asyncio
from typing import Dict, List, Any, Optional, AsyncIterator
from agent import Agent
from config import settings
from runtime_config import runtime_config
from search.hybrid_retriever import HybridRetriever
from search.txt_hybrid_search import TxtHybridSearch
from search.ptf_search import ptf_search

class RagManager:
    """RAG功能管理器 - 处理文件、检索和RAG对话流程"""
    
    # 会话元数据存储路径，指向 Backend/storage/sessions_meta
    SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "sessions_meta")

    def __init__(self, agent: Agent):
        self.agent = agent
        self.retrievers_cache: Dict[str, Any] = {}  # 缓存检索器
        self.expiry_hours = 3
        os.makedirs(self.SESSIONS_DIR, exist_ok=True)

    async def _detect_intent(self, user_input: str, summaries: Dict[str, str]) -> str:
        """
        根据用户输入和文件摘要判断意图
        
        Args:
            user_input: 用户输入
            summaries: 文件摘要字典
            
        Returns:
            'RAG' (需要检索) or 'CHAT' (不需要检索)
        """
        try:
            summary_text = "\n".join([f"文件{i+1}概况: {s[:200]}..." for i, s in enumerate(summaries.values())])
            
            prompt = f"""
            用户正在与一个智能助手对话。用户上传了以下文件，其概况如下：
            
            {summary_text}
            
            用户的最新输入是："{user_input}"
            
            请判断用户的输入是否与上述文件内容有关，或者是否需要查阅上述文件才能回答。
            - 如果用户的问题明显是在询问上述文件的具体内容、细节、或者基于文件内容的分析，请返回 "RAG"。
            - 如果用户的问题是通用的闲聊、问候、或者与上述文件内容完全无关的通用知识问题，请返回 "CHAT"。
            
            请只返回 "RAG" 或 "CHAT"，不要返回其他任何内容。
            """
            
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=prompt)]
            resp = await self.agent.llm.ainvoke(messages)
            intent = resp.content.strip().upper()
            if "RAG" in intent:
                return "RAG"
            return "CHAT"
        except Exception as e:
            print(f"Intent detection failed: {e}")
            return "RAG" # Fallback to RAG just in case

    def _should_net_search(self, user_input: str, enable_net_search: bool) -> bool:
        """判断是否需要联网搜索"""
        try:
            if enable_net_search:
                return True
            q = (user_input or "").lower()
            time_keywords = ["最新", "今年", "最近", "2024", "当前", "update", "today", "now"]
            realtime_keywords = ["股价", "股票", "天气", "news", "新闻", "政策", "price", "trending"]
            if any(k in q for k in time_keywords):
                return True
            if any(k in q for k in realtime_keywords):
                return True
            return False
        except Exception:
            return enable_net_search

    def _should_doc_analysis(self, user_input: str) -> bool:
        """判断是否需要文档整体分析"""
        try:
            q = (user_input or "").strip()
            triggers = ["这是什么", "是什么", "概览", "总体分析", "总结一下", "总结", "介绍一下", "overview"]
            return any(t in q for t in triggers)
        except Exception:
            return False

    async def _detect_search_intent(self, user_input: str, summaries: Dict[str, str] = None) -> dict:
        """
        Detects the user's intent for file search.
        
        Args:
            user_input: 用户输入
            summaries: 文件摘要
            
        Returns:
            A dict with 'type' (SUMMARY/SEARCH/CHAT) and 'query' (optimized search query).
        """
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            summary_info = ""
            if summaries:
                summary_texts = []
                for i, (fp, s) in enumerate(summaries.items(), 1):
                    fname = os.path.basename(fp)
                    # Truncate summary to avoid too long prompt
                    s_trunc = s[:300] + "..." if len(s) > 300 else s
                    summary_texts.append(f"{i}. {fname}: {s_trunc}")
                summary_info = "\n".join(summary_texts)

            prompt = f"""
            User Input: "{user_input}"
            
            Uploaded Documents (Filename: Summary):
            {summary_info}
            
            Analyze the user's intent regarding the uploaded documents.
            1. If the user's input is completely irrelevant to the uploaded documents (e.g. general chat, greetings, questions about unrelated topics), return type "CHAT".
            2. If the user asks for a summary, overview, or general explanation of the file(s), return type "SUMMARY".
            3. If the user asks for specific details, facts, numbers, or answers contained in the file, return type "SEARCH" and provide an optimized search query.
            
            Crucially, identify which specific files are most relevant to the user's query based on the summaries. 
            - If the user asks about "Chinese questions", select only the file related to Chinese.
            - If the user asks about "Math questions", select only the file related to Math.
            - If the query implies searching all files, return an empty list or all filenames.
            
            Return JSON format:
            {{
                "type": "SUMMARY" or "SEARCH" or "CHAT",
                "query": "optimized search query if SEARCH, else empty string",
                "target_files": ["filename1.txt", "filename2.pdf"] 
            }}
            """
            
            messages = [
                SystemMessage(content="You are an intent classifier for a RAG system. Return only JSON."),
                HumanMessage(content=prompt)
            ]
            
            resp = await self.agent.llm.ainvoke(messages)
            content = resp.content.strip()
            # Handle potential code block wrapping
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            return json.loads(content)
        except Exception as e:
            print(f"Intent detection failed: {e}")
            return {"type": "SEARCH", "query": user_input} # Fallback

    async def chat_with_rag_stream(self, session_id: str, user_input: str, enable_net_search: bool = False, file_paths: Optional[List[str]] = None) -> AsyncIterator[str]:
        print(f"[Debug] chat_with_rag_stream called for session {session_id}, input: {user_input[:50]}...", flush=True)
        try:
            auto_paths = None
            if not file_paths:
                auto_paths = self._auto_select_files(session_id, user_input)
            use_paths = file_paths or auto_paths
            retriever = await self.get_or_create_retriever(session_id, file_paths=use_paths, ephemeral=bool(use_paths))

            cache_entry = self.retrievers_cache.get(session_id, {})
            has_docs = bool(cache_entry.get('has_documents', False))
            summaries = cache_entry.get('file_summaries', {})
            
            # Prepare File Context (Agent gets processed file info)
            file_context_str = ""
            if has_docs:
                file_names = [os.path.basename(f) for f in summaries.keys()]
                if file_names:
                    file_context_str = f"【当前已加载文件】: {', '.join(file_names)}\n"

            need_net = self._should_net_search(user_input, enable_net_search)

            # Route A: No Docs, No Net -> Pure Chat
            if not has_docs and not need_net:
                async for chunk in self.agent.chat_stream(session_id=session_id, user_input=user_input, enable_search_tool=False):
                    yield chunk
                return

            # Route B: Docs, No Net -> Document Analysis / RAG
            if has_docs and not need_net:
                # Use LLM to detect intent and optimize query
                intent_res = await self._detect_search_intent(user_input, summaries)
                intent_type = intent_res.get("type", "SEARCH")
                search_query = intent_res.get("query") or user_input
                
                if intent_type == "CHAT":
                    # Irrelevant to docs -> pure chat (but keep file awareness in system context if needed, 
                    # or just pure chat. User said "reply directly". 
                    # Providing file_context_str is harmless and good for "context awareness" even if not retrieving.)
                    async for chunk in self.agent.chat_stream(session_id=session_id, user_input=file_context_str + user_input, enable_search_tool=False):
                        yield chunk
                    return

                if intent_type == "SUMMARY":
                    summary_texts = []
                    for i, (fp, s) in enumerate(summaries.items(), 1):
                        if s:
                            summary_texts.append(f"【文件概览{i} - {os.path.basename(fp)}】\n{s}")
                    summary_block = "\n\n".join(summary_texts) if summary_texts else "暂无文件摘要。"
                    analysis_input = settings.context_wrapper.format(context=summary_block, question=user_input)
                    
                    # Inject file info
                    final_input = file_context_str + analysis_input
                    
                    async for chunk in self.agent.chat_stream(session_id=session_id, user_input=final_input, enable_search_tool=False):
                        yield chunk
                    return

                # Default to SEARCH (Retrieval)
                target_filenames = intent_res.get("target_files", [])
                search_files = []
                if target_filenames:
                    name_map = {os.path.basename(p): p for p in summaries.keys()}
                    for name in target_filenames:
                        if name in name_map:
                            search_files.append(name_map[name])

                print(f"[Debug] Performing search with query: {search_query}, targets: {search_files}")
                retrieved_docs = retriever.search(search_query, k=5, file_paths=search_files if search_files else None) if retriever else []
                if retrieved_docs:
                    contexts = []
                    for i, doc in enumerate(retrieved_docs[:5], 1):
                        content = doc.page_content
                        source_info = f"来源: {doc.metadata.get('source', '未知')}"
                        content_type = doc.metadata.get('type', '文本')
                        source_info += f", 类型: {content_type}"
                        if len(content) > 800:
                            content = content[:800] + "..."
                        contexts.append(f"【参考信息{i}】\n{content}\n{source_info}")
                    context_str = "\n\n".join(contexts)
                    formatted_input = settings.context_wrapper.format(context=context_str, question=user_input)
                    
                    # Inject file info
                    final_input = file_context_str + formatted_input
                    
                    async for chunk in self.agent.chat_stream(session_id=session_id, user_input=final_input, enable_search_tool=False):
                        yield chunk
                    return
                
                # Fallback if no docs found
                async for chunk in self.agent.chat_stream(session_id=session_id, user_input=file_context_str + user_input, enable_search_tool=False):
                    yield chunk
                return

            # Route C: No Docs, Net -> Net Search
            if not has_docs and need_net:
                if runtime_config and not runtime_config.enable_mcp_access:
                    yield "⚠️ **功能受限**：联网搜索功能已被管理员禁用。"
                    return
                available_tools = self.agent.get_available_tools()
                if not available_tools:
                    await self.agent._load_tools_async()
                    available_tools = self.agent.get_available_tools()
                    if not available_tools:
                        yield "⚠️ **搜索服务不可用**：无法连接到联网搜索服务 (MCP工具加载失败)。请联系管理员检查后端日志。"
                        return
                async for chunk in self.agent.chat_stream(session_id=session_id, user_input=user_input, enable_search_tool=True):
                    yield chunk
                return

            # Route D: Docs, Net -> Hybrid
            if has_docs and need_net:
                # Intent analysis for hybrid too?
                # Usually hybrid means we want both. Let's do search first.
                intent_res = await self._detect_search_intent(user_input, summaries)
                intent_type = intent_res.get("type", "SEARCH")
                search_query = intent_res.get("query") or user_input
                
                if intent_type == "CHAT":
                     # Irrelevant to docs -> pure chat (but allow net search)
                     # Using file_context_str so agent knows files exist but skipping retrieval.
                     async for chunk in self.agent.chat_stream(session_id=session_id, user_input=file_context_str + user_input, enable_search_tool=True):
                         yield chunk
                     return

                if intent_type == "SUMMARY":
                    # If user wants summary + net search (rare but possible, e.g. "Summarize this and find latest news on topic")
                    # We can do summary injection + enable_search_tool=True
                    summary_texts = []
                    for i, (fp, s) in enumerate(summaries.items(), 1):
                        if s:
                            summary_texts.append(f"【文件概览{i} - {os.path.basename(fp)}】\n{s}")
                    summary_block = "\n\n".join(summary_texts) if summary_texts else "暂无文件摘要。"
                    analysis_input = settings.context_wrapper.format(context=summary_block, question=user_input)
                    
                    final_input = file_context_str + analysis_input
                    
                    async for chunk in self.agent.chat_stream(session_id=session_id, user_input=final_input, enable_search_tool=True):
                        yield chunk
                    return

                target_filenames = intent_res.get("target_files", [])
                search_files = []
                if target_filenames:
                    name_map = {os.path.basename(p): p for p in summaries.keys()}
                    for name in target_filenames:
                        if name in name_map:
                            search_files.append(name_map[name])

                retrieved_docs = retriever.search(search_query, k=5, file_paths=search_files if search_files else None) if retriever else []
                contexts = []
                for i, doc in enumerate(retrieved_docs[:5], 1):
                    content = doc.page_content
                    source_info = f"来源: {doc.metadata.get('source', '未知')}"
                    content_type = doc.metadata.get('type', '文本')
                    source_info += f", 类型: {content_type}"
                    if len(content) > 800:
                        content = content[:800] + "..."
                    contexts.append(f"【参考信息{i}】\n{content}\n{source_info}")
                context_str = "\n\n".join(contexts)
                hybrid_instruction = user_input + "\n\n(请在回答时结合以上文档内容，并使用联网搜索补充最新信息，引用来源)"
                formatted_input = settings.context_wrapper.format(context=context_str, question=hybrid_instruction)
                
                # Inject file info
                final_input = file_context_str + formatted_input
                
                if runtime_config and not runtime_config.enable_mcp_access:
                    async for chunk in self.agent.chat_stream(session_id=session_id, user_input=final_input, enable_search_tool=False):
                        yield chunk
                    return
                available_tools = self.agent.get_available_tools()
                if not available_tools:
                    await self.agent._load_tools_async()
                async for chunk in self.agent.chat_stream(session_id=session_id, user_input=final_input, enable_search_tool=True):
                    yield chunk
                return

        except Exception as e:
            yield f"处理失败: {str(e)}"

    def _save_session_state(self, session_id: str):
        """保存会话状态到磁盘"""
        try:
            cache_entry = self.retrievers_cache.get(session_id)
            if not cache_entry:
                return
            
            files = cache_entry.get('files', {})
            # Convert sets to lists for JSON serialization
            serializable_files = {
                k: list(v) for k, v in files.items()
            }
            
            data = {
                "session_id": session_id,
                "last_used": cache_entry.get('last_used'),
                "has_documents": cache_entry.get('has_documents'),
                "files": serializable_files,
                "file_summaries": cache_entry.get('file_summaries', {})
            }
            
            file_path = os.path.join(self.SESSIONS_DIR, f"{session_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving session state: {e}")

    def _load_session_state(self, session_id: str) -> bool:
        """从磁盘加载会话状态"""
        try:
            file_path = os.path.join(self.SESSIONS_DIR, f"{session_id}.json")
            if not os.path.exists(file_path):
                return False
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            files = data.get("files", {})
            # Convert lists back to sets
            restored_files = {
                k: set(v) for k, v in files.items()
            }
            
            self.retrievers_cache[session_id] = {
                'retriever': None, # Will be rebuilt on demand
                'last_used': data.get("last_used", time.time()),
                'has_documents': data.get("has_documents", False),
                'files': restored_files,
                'file_summaries': data.get("file_summaries", {})
            }
            return True
        except Exception as e:
            print(f"Error loading session state: {e}")
            return False

    def ensure_session(self, session_id: str): 
        """确保会话存在于缓存中""" 
        if session_id not in self.retrievers_cache:
            # Try to load from disk first
            if not self._load_session_state(session_id):
                self.retrievers_cache[session_id] = { 
                    'retriever': None, 
                    'last_used': time.time(), 
                    'has_documents': False, 
                    'files': {'txt': set(), 'ptf': set(), 'image': set()},
                    'file_summaries': {}
                } 
 
    async def add_document(self, session_id: str, file_path: str, skip_summary: bool = False) -> List: 
        """ 
        添加文档到会话 
         
        Args: 
            session_id: 会话ID 
            file_path: 文件路径 
            skip_summary: 是否跳过自动摘要生成
             
        Returns: 
            操作结果列表 
        """ 
        if not os.path.exists(file_path): 
            return [{"status": "failed", "file": file_path, "message": "文件不存在"}] 
 
        try: 
            # 更新文件记录 
            file_type = self._detect_file_type(file_path) 
            cache_entry = self.retrievers_cache.get(session_id) 
            if not cache_entry: 
                self.retrievers_cache[session_id] = { 
                    'retriever': None, 
                    'last_used': time.time(), 
                    'has_documents': False, 
                    'files': {'txt': set(), 'ptf': set(), 'image': set()},
                    'file_summaries': {}
                } 
                cache_entry = self.retrievers_cache[session_id] 
 
            if file_type == 'txt':
                cache_entry['files']['txt'].add(file_path)
            elif file_type in ['ptf', 'image']:
                if file_type == 'image':
                    cache_entry['files']['image'].add(file_path)
                else:
                    cache_entry['files']['ptf'].add(file_path)
            else:
                return [{"status": "failed", "file": file_path, "message": "不支持的文件类型"}]
 
            # 根据当前文件集合选择或升级检索器 
            retriever = cache_entry.get('retriever') 
            desired = self._desired_retriever_type(cache_entry['files']) 
            retriever = await self._ensure_retriever(session_id, retriever, cache_entry['files'], desired) 
            self.retrievers_cache[session_id]['retriever'] = retriever 
 
            # 添加文件到检索器 
            if retriever and hasattr(retriever, 'add_file'): 
                result = await retriever.add_file(file_path) 
                
                # 兼容不同返回类型
                success = False
                message = "文件添加失败"
                details = {}
                
                if isinstance(result, bool):
                    success = result
                    message = "文件已成功添加" if success else "文件添加失败"
                elif isinstance(result, dict):
                    success = result.get("success", False)
                    message = result.get("message", "文件已成功添加" if success else "文件添加失败")
                    details = result
                
                if success: 
                    self.retrievers_cache[session_id]['last_used'] = time.time() 
                    self.retrievers_cache[session_id]['has_documents'] = True 
                    self._save_session_state(session_id)
                    
                    # Generate Summary if content_preview is available
                    summary_text = ""
                    content_preview = details.get("content_preview", "")
                    if content_preview and not skip_summary:
                        try:
                            print(f"[Debug] Generating summary for content length: {len(content_preview)}", flush=True)
                            from langchain_core.messages import HumanMessage, SystemMessage
                            prompt = f"请简要总结并分析以下文件内容（仅作为概览，不要过长）：\n\n{content_preview[:2000]}"
                            messages = [
                                SystemMessage(content="你是一个智能助手，负责对上传的文件进行简要总结和分析。"),
                                HumanMessage(content=prompt)
                            ]
                            print("[Debug] Invoking LLM for summary...", flush=True)
                            resp = await self.agent.llm.ainvoke(messages)
                            print(f"[Debug] Summary generated: {resp.content[:50]}...", flush=True)
                            summary_text = resp.content
                        except Exception as e:
                            print(f"Summary generation failed: {e}", flush=True)
                            import traceback
                            traceback.print_exc()
                            summary_text = f"（智能总结生成失败: {str(e)}）"
                            
                    # Store summary
                    if summary_text:
                        if 'file_summaries' not in self.retrievers_cache[session_id]:
                             self.retrievers_cache[session_id]['file_summaries'] = {}
                        self.retrievers_cache[session_id]['file_summaries'][file_path] = summary_text
                        self._save_session_state(session_id)

                    return [{
                        "status": "success", 
                        "file": file_path, 
                        "message": message,
                        "details": details,
                        "summary_text": summary_text
                    }] 
            return [{"status": "failed", "file": file_path, "message": "文件添加失败"}] 
 
        except Exception as e: 
            return [{"status": "error", "file": file_path, "message": str(e)}] 
 
    async def get_or_create_retriever(self, session_id: str, file_paths: Optional[List[str]] = None, ephemeral: bool = False) -> Any:
        """ 
        获取或创建检索器 
         
        Args: 
            session_id: 会话ID 
            file_paths: 文件路径列表 
             
        Returns: 
            检索器实例 
        """ 
        if ephemeral and file_paths:
            files_record = {'txt': set(), 'ptf': set(), 'image': set()}
            for fp in file_paths:
                ft = self._detect_file_type(fp)
                if ft == 'txt':
                    files_record['txt'].add(fp)
                elif ft in ['ptf', 'image']:
                    if ft == 'image':
                        files_record['image'].add(fp)
                    else:
                        files_record['ptf'].add(fp)
            desired = self._desired_retriever_type(files_record)
            return await self._ensure_retriever(session_id, None, files_record, desired)

        # 如果缓存中有，优先尝试升级或复用 
        if session_id in self.retrievers_cache: 
            cache_entry = self.retrievers_cache[session_id] 
            retriever = cache_entry.get('retriever') 
            cache_entry['last_used'] = time.time() 
        else:
            # Try loading from disk
            if self._load_session_state(session_id):
                cache_entry = self.retrievers_cache[session_id]
                retriever = None # Loaded state has no retriever instance yet
                cache_entry['last_used'] = time.time()
            else:
                # Will be handled by "new retriever" logic below if file_paths provided,
                # or empty retriever logic at end.
                cache_entry = None
                retriever = None

        if cache_entry:
            # 当已有缓存为空检索器或无法添加文件时，且提供了文件路径，则重建混合检索器 
            need_rebuild = False 
            
            # If retriever is None (loaded from disk or initialized empty), we need to build it
            # based on files in cache_entry
            if retriever is None and (cache_entry['files']['txt'] or cache_entry['files']['ptf'] or cache_entry['files']['image']):
                need_rebuild = True
            
            if file_paths and file_paths: 
                if retriever and not hasattr(retriever, 'add_file'): 
                    need_rebuild = True 
                elif retriever: 
                    try: 
                        # 尝试向现有检索器添加文件；若失败则触发重建 
                        for file_path in file_paths: 
                            added = await retriever.add_file(file_path) 
                            if added: 
                                cache_entry['has_documents'] = True 
                            else: 
                                need_rebuild = True 
                    except Exception: 
                        need_rebuild = True 

            if need_rebuild: 
                try: 
                    # 更新文件集合记录 
                    files_record = cache_entry.get('files') or {'txt': set(), 'ptf': set(), 'image': set()} 
                    for fp in (file_paths or []): 
                        ft = self._detect_file_type(fp) 
                        if ft == 'txt': 
                            files_record['txt'].add(fp) 
                        elif ft in ['ptf', 'image']: 
                            if ft == 'image': 
                                files_record['image'].add(fp) 
                            else: 
                                files_record['ptf'].add(fp) 
                    cache_entry['files'] = files_record 

                    desired = self._desired_retriever_type(files_record) 
                    new_retriever = await self._ensure_retriever(session_id, retriever, files_record, desired) 
                    if new_retriever: 
                        self.retrievers_cache[session_id]['retriever'] = new_retriever 
                        self.retrievers_cache[session_id]['last_used'] = time.time() 
                        self.retrievers_cache[session_id]['has_documents'] = True 
                        self._save_session_state(session_id)
                        return new_retriever 
                except Exception: 
                    pass 

            if retriever:
                return retriever

        # 创建新的检索器（首次有文件时） 
        retriever = None 
        if file_paths and len(file_paths) > 0: 
            # 构建文件集合记录 
            files_record = {'txt': set(), 'ptf': set(), 'image': set()} 
            for fp in file_paths: 
                ft = self._detect_file_type(fp) 
                if ft == 'txt': 
                    files_record['txt'].add(fp) 
                elif ft in ['ptf', 'image']: 
                    if ft == 'image': 
                        files_record['image'].add(fp) 
                    else: 
                        files_record['ptf'].add(fp) 
            desired = self._desired_retriever_type(files_record) 
            retriever = await self._ensure_retriever(session_id, None, files_record, desired) 

        if retriever: 
            self.retrievers_cache[session_id] = { 
                'retriever': retriever, 
                'last_used': time.time(), 
                'has_documents': bool(file_paths), 
                'files': {'txt': files_record['txt'], 'ptf': files_record['ptf'], 'image': files_record['image']} 
            } 
            self._save_session_state(session_id)
        elif not file_paths: 
            # 没有文件时创建空检索器（仅用于RAG判断） 
            class EmptyRetriever: 
                def search(self, query: str, k: int = 3): 
                    return [] 

                async def add_file(self, file_path: str) -> bool: 
                    # 空检索器无法添加文件 
                    return False 

            retriever = EmptyRetriever() 
            self.retrievers_cache[session_id] = { 
                'retriever': retriever, 
                'last_used': time.time(), 
                'has_documents': False 
            } 

        return retriever 

    def _detect_file_type(self, file_path: str) -> str: 
        ext = os.path.splitext(file_path)[1].lower() 
        if ext in ['.txt', '.md']: 
            return 'txt' 
        elif ext in ['.ptf', '.pdf', '.docx', '.pptx']: 
            return 'ptf' 
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif']: 
            return 'image' 
        else: 
            return 'unknown' 
 
    def _desired_retriever_type(self, files_record: Dict[str, set]) -> str: 
        has_txt = bool(files_record.get('txt')) 
        has_ptf = bool(files_record.get('ptf') or files_record.get('image')) 
        if has_txt and has_ptf: 
            return 'hybrid' 
        elif has_txt: 
            return 'txt' 
        elif has_ptf: 
            return 'ptf' 
        return 'empty' 
 
    async def _ensure_retriever(self, session_id: str, current, files_record: Dict[str, set], desired: str): 
        try: 
            if desired == 'empty': 
                class EmptyRetriever: 
                    def search(self, query: str, k: int = 3): 
                        return [] 
                    async def add_file(self, file_path: str) -> bool: 
                        return False 
                return EmptyRetriever() 
            if desired == 'txt': 
                if isinstance(current, TxtHybridSearch): 
                    return current 
                txt = TxtHybridSearch(session_id) 
                for fp in files_record.get('txt', []): 
                    try: 
                        await txt.process_and_index(fp) 
                    except Exception: 
                        continue 
                return txt 
            if desired == 'ptf': 
                if isinstance(current, ptf_search): 
                    return current 
                ptf = ptf_search(session_id) 
                for fp in (files_record.get('ptf', set()) | files_record.get('image', set())): 
                    try: 
                        await ptf.process_and_embed(fp) 
                    except Exception: 
                        continue 
                return ptf 
            if desired == 'hybrid': 
                if isinstance(current, HybridRetriever): 
                    return current 
                # 复用或新建子检索器 
                txt = TxtHybridSearch(session_id) 
                for fp in files_record.get('txt', []): 
                    try: 
                        await txt.process_and_index(fp) 
                    except Exception: 
                        continue 
                ptf = ptf_search(session_id) 
                for fp in (files_record.get('ptf', set()) | files_record.get('image', set())): 
                    try: 
                        await ptf.process_and_embed(fp) 
                    except Exception: 
                        continue 
                return HybridRetriever(txt, ptf) 
        except Exception: 
            return current 

    def _auto_select_files(self, session_id: str, user_input: str) -> Optional[List[str]]:
        try:
            cache_entry = self.retrievers_cache.get(session_id)
            if not cache_entry:
                if not self._load_session_state(session_id):
                    return None
                cache_entry = self.retrievers_cache.get(session_id)

            files_record = cache_entry.get('files') or {'txt': set(), 'ptf': set(), 'image': set()}
            all_files = list(files_record.get('txt', set()) | files_record.get('ptf', set()) | files_record.get('image', set()))
            if not all_files:
                return None

            q = (user_input or '').lower()
            selected = []
            for fp in all_files:
                name = os.path.basename(fp).lower()
                base, _ext = os.path.splitext(name)
                if base and base in q:
                    selected.append(fp)
                elif name in q:
                    selected.append(fp)
            # 如果自动选择命中至少一个文件且不是全量，返回所选文件；否则返回None代表使用全量
            if selected and len(selected) < len(all_files):
                return selected
            return None
        except Exception:
            return None

    async def chat_with_rag(self, session_id: str, user_input: str, enable_net_search: bool = False, file_paths: Optional[List[str]] = None) -> str:
        try:
            auto_paths = None
            if not file_paths:
                auto_paths = self._auto_select_files(session_id, user_input)
            use_paths = file_paths or auto_paths
            retriever = await self.get_or_create_retriever(session_id, file_paths=use_paths, ephemeral=bool(use_paths))

            cache_entry = self.retrievers_cache.get(session_id, {})
            has_docs = bool(cache_entry.get('has_documents', False))
            summaries = cache_entry.get('file_summaries', {})
            
            # Prepare File Context
            file_context_str = ""
            if has_docs:
                file_infos = []
                for fp in summaries.keys():
                    fname = os.path.basename(fp)
                    s = summaries.get(fp, "")
                    if s:
                        # Limit summary length to avoid excessive context
                        s_short = s[:200].replace("\n", " ") + "..." if len(s) > 200 else s.replace("\n", " ")
                        file_infos.append(f"{fname}: {s_short}")
                    else:
                        file_infos.append(fname)
                
                if file_infos:
                    file_context_str = "【当前已加载文件】:\n" + "\n".join([f"- {info}" for info in file_infos]) + "\n"

            need_net = self._should_net_search(user_input, enable_net_search)

            if not has_docs and not need_net:
                return await self.agent.chat(session_id=session_id, user_input=user_input)

            if has_docs and not need_net:
                # Use LLM to detect intent and optimize query
                intent_res = await self._detect_search_intent(user_input, summaries)
                intent_type = intent_res.get("type", "SEARCH")
                search_query = intent_res.get("query") or user_input

                if intent_type == "CHAT":
                    # Irrelevant to docs -> pure chat (but keep file awareness in system context)
                    return await self.agent.chat(session_id=session_id, user_input=file_context_str + user_input)

                if intent_type == "SUMMARY":
                    summary_texts = []
                    for i, (fp, s) in enumerate(summaries.items(), 1):
                        if s:
                            summary_texts.append(f"【文件概览{i} - {os.path.basename(fp)}】\n{s}")
                    summary_block = "\n\n".join(summary_texts) if summary_texts else "暂无文件摘要。"
                    analysis_input = settings.context_wrapper.format(context=summary_block, question=user_input)
                    return await self.agent.chat(session_id=session_id, user_input=file_context_str + analysis_input)

                # SEARCH
                retrieved_docs = retriever.search(search_query, k=5) if retriever else []
                if retrieved_docs:
                    contexts = []
                    for i, doc in enumerate(retrieved_docs[:5], 1):
                        content = doc.page_content
                        source_info = f"来源: {doc.metadata.get('source', '未知')}"
                        content_type = doc.metadata.get('type', '文本')
                        source_info += f", 类型: {content_type}"
                        if len(content) > 800:
                            content = content[:800] + "..."
                        contexts.append(f"【参考信息{i}】\n{content}\n{source_info}")
                    context_str = "\n\n".join(contexts)
                    formatted_input = settings.context_wrapper.format(context=context_str, question=user_input)
                    return await self.agent.chat(session_id=session_id, user_input=file_context_str + formatted_input)
                
                return await self.agent.chat(session_id=session_id, user_input=file_context_str + user_input)

            if not has_docs and need_net:
                if runtime_config and not runtime_config.enable_mcp_access:
                    return "⚠️ **功能受限**：联网搜索功能已被管理员禁用。"
                available_tools = self.agent.get_available_tools()
                if not available_tools:
                    await self.agent._load_tools_async()
                    available_tools = self.agent.get_available_tools()
                    if not available_tools:
                        return "⚠️ **搜索服务不可用**：无法连接到联网搜索服务 (MCP工具加载失败)。请联系管理员检查后端日志。"
                search_input = user_input
                return await self.agent.chat(session_id=session_id, user_input=search_input)

            if has_docs and need_net:
                # Hybrid Intent
                intent_res = await self._detect_search_intent(user_input, summaries)
                intent_type = intent_res.get("type", "SEARCH")
                search_query = intent_res.get("query") or user_input
                
                if intent_type == "CHAT":
                    # Irrelevant to docs -> pure chat (but allow net search)
                    search_input = user_input
                    if runtime_config and not runtime_config.enable_mcp_access:
                         return await self.agent.chat(session_id=session_id, user_input=file_context_str + search_input)
                    available_tools = self.agent.get_available_tools()
                    if not available_tools:
                         await self.agent._load_tools_async()
                    return await self.agent.chat(session_id=session_id, user_input=file_context_str + search_input)

                if intent_type == "SUMMARY":
                    # Summary + Net Search
                    summary_texts = []
                    for i, (fp, s) in enumerate(summaries.items(), 1):
                        if s:
                            summary_texts.append(f"【文件概览{i} - {os.path.basename(fp)}】\n{s}")
                    summary_block = "\n\n".join(summary_texts) if summary_texts else "暂无文件摘要。"
                    analysis_input = settings.context_wrapper.format(context=summary_block, question=user_input)
                    final_input = file_context_str + analysis_input
                    
                    if runtime_config and not runtime_config.enable_mcp_access:
                        return await self.agent.chat(session_id=session_id, user_input=final_input)
                    available_tools = self.agent.get_available_tools()
                    if not available_tools:
                        await self.agent._load_tools_async()
                    return await self.agent.chat(session_id=session_id, user_input=final_input)

                retrieved_docs = retriever.search(search_query, k=5) if retriever else []
                contexts = []
                for i, doc in enumerate(retrieved_docs[:5], 1):
                    content = doc.page_content
                    source_info = f"来源: {doc.metadata.get('source', '未知')}"
                    content_type = doc.metadata.get('type', '文本')
                    source_info += f", 类型: {content_type}"
                    if len(content) > 800:
                        content = content[:800] + "..."
                    contexts.append(f"【参考信息{i}】\n{content}\n{source_info}")
                context_str = "\n\n".join(contexts)
                hybrid_instruction = user_input + "\n\n(请在回答时结合以上文档内容，并使用联网搜索补充最新信息，引用来源)"
                formatted_input = settings.context_wrapper.format(context=context_str, question=hybrid_instruction)
                if runtime_config and not runtime_config.enable_mcp_access:
                    return await self.agent.chat(session_id=session_id, user_input=file_context_str + formatted_input)
                available_tools = self.agent.get_available_tools()
                if not available_tools:
                    await self.agent._load_tools_async()
                return await self.agent.chat(session_id=session_id, user_input=file_context_str + formatted_input)

            return await self.agent.chat(session_id=session_id, user_input=user_input)
        except Exception as e:
            return f"处理失败: {str(e)}"
 
    def cleanup_session_retriever(self, session_id: str): 
        """清理会话的检索器资源""" 
        if session_id in self.retrievers_cache: 
            retriever_info = self.retrievers_cache[session_id] 
            retriever = retriever_info.get('retriever') 
            if hasattr(retriever, 'cleanup'): 
                try: 
                    retriever.cleanup() 
                except Exception: 
                    pass 
            del self.retrievers_cache[session_id] 
 
    def cleanup_expired_retrievers(self) -> List[str]: 
        """清理过期的检索器，返回被清理的会话ID列表""" 
        sessions_to_clean = [] 
        current_time = time.time() 
 
        for session_id, retriever_info in list(self.retrievers_cache.items()): 
            last_used = retriever_info.get('last_used', 0) 
            if (current_time - last_used) / 3600 > self.expiry_hours: 
                sessions_to_clean.append(session_id) 
 
        for session_id in sessions_to_clean: 
            self.cleanup_session_retriever(session_id) 
 
        return sessions_to_clean 
 
    def get_retriever_info(self, session_id: str) -> dict: 
        """获取检索器状态信息""" 
        if session_id in self.retrievers_cache: 
            cache_entry = self.retrievers_cache[session_id] 
            retriever = cache_entry['retriever'] 
 
            files_record = cache_entry.get('files') or {'txt': set(), 'ptf': set(), 'image': set()} 
            documents = sorted(list(files_record.get('txt', set()) | files_record.get('ptf', set()) | files_record.get('image', set()))) 
 
            info = { 
                "has_retriever": True, 
                "has_documents": cache_entry.get('has_documents', False), 
                "last_used": time.ctime(cache_entry['last_used']), 
                "idle_hours": round((time.time() - cache_entry['last_used']) / 3600, 2), 
                "documents": documents 
            } 
 
            if hasattr(retriever, 'get_stats'): 
                try: 
                    info['retriever_stats'] = retriever.get_stats() 
                except Exception: 
                    pass 
 
            return info 
 
        return { 
            "has_retriever": False, 
            "has_documents": False, 
            "documents": [] 
        } 
