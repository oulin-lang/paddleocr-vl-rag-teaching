import os
import time
import json
import asyncio
from typing import List, Dict, Optional, AsyncIterator, Any
from tenacity import retry, stop_after_attempt, wait_fixed, stop_after_delay, retry_if_exception_type


try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return None

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableLambda

from config import settings
from runtime_config import runtime_config

load_dotenv()


class Agent:
    """智能体类 - 统一的高质量教师辅助智能体"""

    def __init__(self, model_name: str = settings.model_name):
        """
        初始化智能体

        Args:
            model_name: 模型名称
        """
        # 初始化大语言模型
        self.llm = ChatOpenAI(
            api_key=os.getenv('ALIYUNBAILIAN_API_KEY'),
            base_url=settings.base_url,
            model=model_name,
            temperature=0.7,
            max_tokens=2000,
            streaming=True  # 启用流式
        )

        # 存储会话历史
        self.session_memory: Dict[str, List[Dict]] = {}
        # 会话记忆存储路径，指向 Backend/storage/session_memory
        self.memory_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "session_memory")
        os.makedirs(self.memory_dir, exist_ok=True)

        # 工具列表和MCP客户端
        self.tools = []
        self.agent = None          # 当前使用的Agent
        self.agent_basic = None    # 基础Agent (无工具)
        self.agent_search = None   # 搜索Agent (有工具)
        self.mcp_client = None
        self.checkpointer = InMemorySaver() # 共享检查点

        # 标记初始化状态
        self._initialized = False

    async def _create_agent(self, tools: List[Any]):
        """
        创建智能体（通用方法）
        
        Args:
            tools: 工具列表
            
        Returns:
            创建的智能体实例
        """
        try:
            # 使用LangChain 1.x的create_agent函数
            agent = create_agent(
                model=self.llm,
                tools=tools,
                system_prompt=settings.system_prompt,
                checkpointer=self.checkpointer,
            )
            return agent
        except Exception:
            return await self._create_fallback_agent() 
 
    async def _create_fallback_agent(self): 
        """ 
        创建回退基础智能体（异步方法） 
        
        Returns: 
            基础智能体实例 
        """ 
        from langchain_core.prompts import ChatPromptTemplate
        
        # 使用PromptTemplate + LLM构建简单的Chain，支持流式输出
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{input}"),
        ])
        
        self.agent = prompt | self.llm
        return self.agent 
 
    async def chat(self, session_id: str, user_input: str, system_prompt: Optional[str] = None) -> str: 
        """ 
        统一的聊天方法（异步版本） 
 
        Args: 
            session_id: 会话ID 
            user_input: 用户输入 
            system_prompt: 系统提示词 (可选，通常不再需要，使用统一Prompt) 
 
        Returns: 
            智能体响应 
        """ 
        try: 
            # 确保智能体已初始化 
            if not self._initialized: 
                await self._async_init() 
                self._initialized = True 
 
            # 默认使用统一配置的system_prompt，除非明确覆盖 
            if system_prompt is None: 
                system_prompt = settings.system_prompt 
 
            # 更新会话记忆 
            if session_id not in self.session_memory: 
                self.session_memory[session_id] = [] 
 
            self.session_memory[session_id].append({ 
                "role": "user", 
                "content": user_input, 
                "timestamp": time.time() 
            }) 
 
            # 准备输入数据 
            input_data = { 
                "input": user_input, 
                "messages": [HumanMessage(content=user_input)], 
                "system_prompt": system_prompt 
            } 
 
            # 配置参数（包含检查点线程ID） 
            config = { 
                "configurable": { 
                    "thread_id": session_id 
                } 
            } 

            # 调用智能体（异步调用） 
            result = await self._async_chat(input_data, config) 
            if not result: 
                # 尝试同步调用 
                result = self.agent.invoke(input_data, config=config) if self.agent else None 

            # 提取回复内容 
            response = self._extract_response(result) if result else "未收到响应" 
 
            # 保存到记忆 
            self.session_memory[session_id].append({ 
                "role": "assistant", 
                "content": response, 
                "timestamp": time.time() 
            }) 
 
            # 保存会话到文件 
            self._save_session(session_id) 
 
            return response 
 
        except Exception as e: 
            import traceback
            traceback.print_exc()
            error_msg = f"处理失败: {str(e)}" 
            return error_msg 
 
    async def _async_chat(self, input_data: Dict, config: Dict): 
        """ 
        异步聊天方法 
        
        Args:
            input_data: 输入数据
            config: 运行配置
            
        Returns:
            调用结果
        """ 
        if self.agent: 
            try: 
                # 尝试异步调用 
                if hasattr(self.agent, 'ainvoke'): 
                    result = await self.agent.ainvoke(input_data, config=config) 
                else: 
                    # 降级到同步调用 
                    result = self.agent.invoke(input_data, config=config) 
                return result 
            except Exception: 
                import traceback
                traceback.print_exc()
                return None 
        return None 
 
    def _extract_response(self, result) -> str: 
        """ 
        提取智能体响应 
        
        Args:
            result: 智能体执行结果
            
        Returns:
            提取的回复内容
        """ 
        response_content = ""
        if isinstance(result, dict): 
            # 检查是否有messages字段 
            if "messages" in result: 
                messages = result["messages"] 
                if messages: 
                    last_message = messages[-1] 
                    if hasattr(last_message, 'content'): 
                        response_content = last_message.content 
                    elif isinstance(last_message, dict): 
                        response_content = last_message.get("content", str(last_message)) 
            
            # 检查是否有output字段 (如果messages没找到或为空)
            if not response_content and "output" in result: 
                output = result["output"] 
                if hasattr(output, 'content'): 
                    response_content = output.content 
                else: 
                    response_content = str(output) 
        else:
            # 默认返回字符串表示 
            response_content = str(result)
        
        # 处理回复内容中的格式问题
        if response_content:
            # 替换字面量的 "/n" 为换行符 (针对用户反馈的问题)
            response_content = response_content.replace("/n", "\n")
            # 替换可能的转义换行符 "\\n" 为 "\n"
            response_content = response_content.replace("\\n", "\n")
            
        return response_content 
 
    async def _async_init(self):
        """异步初始化智能体"""
        # 加载工具
        await self._load_tools_async()

        # 创建基础智能体 (无工具)
        self.agent_basic = await self._create_agent(tools=[])

        # 创建搜索智能体 (有工具)
        if self.tools:
            self.agent_search = await self._create_agent(tools=self.tools)
        else:
            self.agent_search = self.agent_basic

        # 默认使用基础智能体
        self.agent = self.agent_basic
        self._initialized = True

    async def _load_tools_async(self):
        """异步加载工具 (带重试机制)"""
        self.tools = []
        
        # 1. 尝试加载 MCP 工具
        if runtime_config.enable_mcp_access and settings.mcp_servers:
            try:
                self.mcp_client = MultiServerMCPClient(settings.mcp_servers)
                mcp_tools = await self.mcp_client.get_tools()
                self.tools.extend(mcp_tools)
                print(f"[Agent] Successfully loaded {len(mcp_tools)} tools from MCP servers.")
            except Exception as e:
                print(f"[Agent] Failed to load MCP tools: {e}")
        
        # 2. 尝试加载 Tavily 搜索工具 (作为备用或补充)
        if os.getenv("TAVILY_API_KEY"):
            try:
                # 包装 Tavily 工具以支持重试
                tavily_tool = TavilySearchResults(max_results=5)
                
                # 暂时移除 invoke 的 monkey patch，因为它导致了 "object has no field 'invoke'" 错误
                # 如果需要重试机制，应该使用 LangChain 的 .with_retry() 方法或者其他标准方式
                
                self.tools.append(tavily_tool)
                print("[Agent] Added TavilySearchResults tool.")
            except Exception as e:
                print(f"[Agent] Failed to add Tavily tool: {e}")
 
 
    async def should_search(self, user_input: str) -> bool:
        """
        判断用户问题是否需要联网搜索
        
        Args:
            user_input: 用户输入
            
        Returns:
            True if search is needed, False otherwise
        """
        try:
            prompt = f"""
            请判断以下用户问题是否需要实时联网搜索才能回答。
            如果是关于时事新闻、天气、最新技术动态、特定具体数据等需要外部信息的问题，返回 "YES"。
            如果是通用知识、逻辑推理、闲聊或已有上下文的问题，返回 "NO"。
            
            问题: {user_input}
            
            只返回 YES 或 NO，不要有其他内容。
            """
            messages = [HumanMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)
            content = response.content.strip().upper()
            return "YES" in content
        except Exception:
            return False

    async def chat_stream(self, session_id: str, user_input: str, system_prompt: Optional[str] = None, enable_search_tool: bool = False) -> AsyncIterator[str]:
        """
        流式聊天方法
        
        Args:
            session_id: 会话ID
            user_input: 用户输入
            system_prompt: 系统提示词
            enable_search_tool: 是否启用搜索工具
            
        Yields:
            流式响应片段
        """
        # 确保智能体已初始化
        if not self._initialized:
            await self._async_init()

        # 默认使用统一配置的system_prompt
        if system_prompt is None:
            system_prompt = settings.system_prompt

        # 更新会话记忆
        if session_id not in self.session_memory:
            self.session_memory[session_id] = []

        self.session_memory[session_id].append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time()
        })

        # 准备输入数据
        input_data = {
            "input": user_input,
            "messages": [HumanMessage(content=user_input)],
            "system_prompt": system_prompt
        }

        # 配置参数
        config = {
            "configurable": {
                "thread_id": session_id
            }
        }

        # 选择Agent
        agent_to_use = self.agent_search if (enable_search_tool and self.agent_search) else self.agent_basic
        
        # 尝试修剪历史记录以限制上下文长度
        try:
            # 检查是否支持 LangGraph 状态管理
            if hasattr(agent_to_use, "aget_state") and hasattr(agent_to_use, "aupdate_state"):
                current_state = await agent_to_use.aget_state(config)
                if current_state and current_state.values and "messages" in current_state.values:
                    messages = current_state.values["messages"]
                    total_chars = 0
                    for m in messages:
                        if hasattr(m, 'content') and isinstance(m.content, str):
                            total_chars += len(m.content)
                    
                    if total_chars > settings.max_history_chars:
                        chars_to_remove = total_chars - settings.max_history_chars
                        removed_chars = 0
                        messages_to_remove = []
                        
                        for m in messages:
                            # 保留 SystemMessage
                            if isinstance(m, SystemMessage):
                                continue
                            
                            # 仅移除具有 ID 的消息
                            if hasattr(m, 'id') and m.id:
                                content_len = len(m.content) if (hasattr(m, 'content') and isinstance(m.content, str)) else 0
                                messages_to_remove.append(RemoveMessage(id=m.id))
                                removed_chars += content_len
                                
                                if removed_chars >= chars_to_remove:
                                    break
                        
                        if messages_to_remove:
                            print(f"[Agent] Trimming history: removing {len(messages_to_remove)} messages to save {removed_chars} chars.")
                            await agent_to_use.aupdate_state(config, {"messages": messages_to_remove})
        except Exception as e:
            print(f"[Agent] Warning: Failed to trim history: {e}")

        full_response = ""
        
        try:
            # 使用 astream_events 获取流式输出
            # version='v1' 兼容性更好
            async for event in agent_to_use.astream_events(input_data, config=config, version="v1"):
                kind = event["event"]
                # print(f"[Debug] Agent event: {kind}", flush=True)
                
                # 过滤并提取文本生成内容
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    print(f"[Debug] Chunk content: {repr(chunk.content)}", flush=True)
                    if hasattr(chunk, "content"):
                        content = chunk.content
                        if content:
                            full_response += content
                            yield content
                
                # 可选: 处理工具调用事件
                if kind == "on_tool_start":
                     # yield f"\n[Thinking: Calling tool {event['name']}...]\n"
                     yield "\n🔍 **正在联网搜索相关信息...**\n"
                
                if kind == "on_tool_end":
                     # yield f"\n[Thinking: Tool execution completed]\n"
                     yield "\n✅ **搜索完成，正在整理回答...**\n"

        except asyncio.CancelledError:
            # 处理流被取消的情况（如客户端断开连接）
            full_response += "\n[Interrupted]"
            print(f"[Agent] Stream cancelled for session {session_id}")
            raise

        except Exception as e:
            error_msg = f"Error during streaming: {str(e)}"
            print(error_msg)
            full_response += f"\n[Error: {str(e)}]"
            yield f"\n[System Error: {str(e)}]"

        finally:
            # 无论成功、失败还是中断，都保存生成的回复到记忆中
            if full_response.strip():
                self.session_memory[session_id].append({
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": time.time()
                })

                # 保存会话到文件
                self._save_session(session_id)


    def _save_session(self, session_id: str): 
        """ 
        保存会话到文件 
        """ 
        try: 
            file_path = os.path.join(self.memory_dir, f"session_{session_id}.json") 
 
            session_data = { 
                "session_id": session_id, 
                "memory": self.session_memory.get(session_id, []), 
                "save_time": time.time(), 
                "save_date": time.strftime("%Y-%m-%d %H:%M:%S") 
            } 
 
            with open(file_path, 'w', encoding='utf-8') as f: 
                json.dump(session_data, f, ensure_ascii=False, indent=2) 
 
        except Exception: 
            pass 
 
    def clear_session(self, session_id: str): 
        """ 
        清理会话记忆 
        """ 
        if session_id in self.session_memory: 
            del self.session_memory[session_id] 
 
        file_path = os.path.join(self.memory_dir, f"session_{session_id}.json") 
        if os.path.exists(file_path): 
            os.remove(file_path) 
 
    def get_available_tools(self) -> List[str]: 
        """ 
        获取可用工具列表 
        """ 
        return [tool.name for tool in self.tools] 
 
    def get_session_history(self, session_id: str) -> List[Dict]: 
        """ 
        获取会话历史 
        """ 
        return self.session_memory.get(session_id, []) 
