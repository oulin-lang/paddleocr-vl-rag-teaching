import os
import time
import numpy as np
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
import shutil
import dashscope
from config import Settings as st
from runtime_config import runtime_config

settings = st()


class CustomDashScopeEmbeddings(Embeddings):
    """自定义DashScope嵌入类，符合LangChain的Embeddings接口"""

    def __init__(self, api_key: str = None, model: str = settings.model_multimodal):
        """
        初始化嵌入器

        Args:
            api_key: DashScope API密钥
            model: 模型名称
        """
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY') or os.getenv('ALIYUNBAILIAN_API_KEY')
        self.model = model
        # 允许API Key为空（因为如果禁用联网，可能不需要Key），但如果有Key则设置
        if self.api_key:
            dashscope.api_key = self.api_key
        # if not self.api_key:
        #     raise ValueError("未设置阿里云API Key")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为文档列表生成嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        print(f"[Debug] CustomDashScopeEmbeddings.embed_documents: {len(texts)} texts", flush=True)
        # 检查联网权限
        if runtime_config and not runtime_config.enable_mcp_access:
            print(f"[Debug] MCP access disabled, using hash vector", flush=True)
            return [self._generate_hash_vector(text) for text in texts]

        all_embeddings = []

        for i, text in enumerate(texts):
            try:
                print(f"[Debug] Embedding text {i+1}/{len(texts)}...", flush=True)
                if not text or not text.strip():
                    all_embeddings.append(self._generate_hash_vector(""))
                    continue

                formatted_inputs = [{'text': text.strip()}]

                print(f"[Debug] Calling dashscope.MultiModalEmbedding.call...", flush=True)
                resp = dashscope.MultiModalEmbedding.call(
                    model=self.model,
                    input=formatted_inputs
                )
                print(f"[Debug] DashScope response code: {resp.status_code}", flush=True)

                if resp.status_code == 200:
                    embedding = resp.output['embeddings'][0]['embedding']
                    all_embeddings.append(embedding)
                else:
                    print(f"[Debug] DashScope failed: {resp.message}", flush=True)
                    all_embeddings.append(self._generate_hash_vector(text))

            except Exception as e:
                print(f"[Debug] Embedding exception: {e}", flush=True)
                all_embeddings.append(self._generate_hash_vector(text))

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        为单个查询生成嵌入向量

        Args:
            text: 查询文本

        Returns:
            嵌入向量
        """
        if not text or not text.strip():
            return self._generate_hash_vector("")

        # 检查联网权限
        if runtime_config and not runtime_config.enable_mcp_access:
            return self._generate_hash_vector(text)

        try:
            formatted_inputs = [{'text': text.strip()}]

            resp = dashscope.MultiModalEmbedding.call(
                model=self.model,
                input=formatted_inputs
            )

            if resp.status_code == 200:
                return resp.output['embeddings'][0]['embedding']
            else:
                return self._generate_hash_vector(text)

        except Exception:
            return self._generate_hash_vector(text)

    def _generate_hash_vector(self, text: str) -> List[float]:
        """
        基于文本哈希的确定性向量（降级方案）

        Returns:
            1024维稳定向量
        """
        try:
            seed = (abs(hash(text)) % (2**32)) if text is not None else 0
        except Exception:
            seed = 0
        rng = np.random.default_rng(seed)
        return rng.standard_normal(1024).astype(float).tolist()


class ptf_search:
    """PDF/图像文档的检索器，使用向量检索"""

    def __init__(self, session_id: str,
                 dashscope_api_key: str = None,
                 embedding_model_name: str = "qwen2.5-vl"):
        """
        初始化检索器

        Args:
            session_id: 会话ID
            dashscope_api_key: 阿里云DashScope API Key
            embedding_model_name: 向量模型标识
        """
        self.session_id = session_id
        self.retriever_type = "vector"
        self.embedding_model_name = embedding_model_name

        try:
            from file_management.ptf_processor import PaddlePTFParser
            # Set output_dir to Backend/storage/uploads so extracted images are accessible via /uploads endpoint
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            uploads_dir = os.path.join(base_dir, "storage", "uploads")
            self.ocr_service = PaddlePTFParser(session_id=session_id, output_dir=uploads_dir)
            self.ocr_available = True
        except Exception:
            self.ocr_service = None
            self.ocr_available = False

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )

        self.embeddings = CustomDashScopeEmbeddings(
            api_key=dashscope_api_key,
            model="qwen2.5-vl-embedding"
        )

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_dir = os.path.join(base_dir, "storage", "session_dbs", session_id)
        self.created_time = time.time()
        self.last_used_time = time.time()
        self.vector_store = self._load_or_create_vector_store()
        self._files: set[str] = set()

    def _load_or_create_vector_store(self):
        try:
            os.makedirs(self.db_dir, exist_ok=True)
            if os.path.exists(os.path.join(self.db_dir, "index.faiss")):
                return FAISS.load_local(self.db_dir, self.embeddings, allow_dangerous_deserialization=True)
            return None
        except Exception:
            return None

    async def process_and_embed(self, file_path: str) -> tuple[List[Document], str, Dict[str, Any]]:
        """
        处理文档并嵌入到向量数据库
        
        Args:
            file_path: 文件路径
            
        Returns:
            (Document对象列表, 全文内容, 结构化数据)
        """
        try:
            print(f"[Debug] ptf_search.process_and_embed: {file_path}", flush=True)
            if not os.path.exists(file_path):
                print(f"[Debug] File not found: {file_path}", flush=True)
                return [], "", {}

            if not self.ocr_service:
                print(f"[Debug] OCR service not available", flush=True)
                return [], "", {}
            
            # 使用 executor 运行同步的 OCR 解析
            import asyncio
            loop = asyncio.get_running_loop()
            print(f"[Debug] Starting OCR parsing via executor...", flush=True)
            result = await loop.run_in_executor(None, self.ocr_service.parse_file, file_path)
            print(f"[Debug] OCR parsing finished. Result type: {type(result)}", flush=True)
            
            documents = []
            structured_data = {}

            if isinstance(result, dict):
                if "structured_data" in result:
                    structured_data = result["structured_data"]
                
                if "elements" in result:
                    elements = result["elements"]
                elif "structured_data" in result:
                    elements = []
                    for page_elements in structured_data.get("elements_by_page", {}).values():
                        elements.extend(page_elements)
                else:
                    elements = []
            elif isinstance(result, list):
                elements = result
            else:
                elements = []

            file_name = os.path.basename(file_path)
            file_type = self._get_file_type(file_path)

            for elem in elements:
                if isinstance(elem, dict):
                    content = elem.get('content', '')
                    elem_type = elem.get('type', 'text')

                    # 标准化元素类型
                    if 'table' in str(elem_type).lower() or '表格' in str(elem_type):
                        elem_type = 'table'
                    elif 'figure' in str(elem_type).lower() or 'image' in str(elem_type).lower() or '图表' in str(
                            elem_type) or '图片' in str(elem_type):
                        elem_type = 'figure'
                    elif 'formula' in str(elem_type).lower() or '公式' in str(elem_type):
                        elem_type = 'formula'

                    if elem_type == 'text' and content:
                        chunks = self.splitter.split_text(str(content))
                        for i, chunk in enumerate(chunks):
                            if chunk.strip():
                                documents.append(Document(
                                    page_content=chunk,
                                    metadata={
                                        "type": elem_type,
                                        "original_type": elem_type,
                                        "source": "text_chunk",
                                        "file_path": file_path,
                                        "session_id": self.session_id,
                                        "retriever_type": self.retriever_type,
                                        "file": file_name,
                                        "page": elem.get('page', 1),
                                        "file_type": file_type,
                                        "embedding_model": self.embedding_model_name,
                                        "chunk_id": f"{file_name}_{elem_type}_{i}"
                                    }
                                ))
                    elif content:
                        documents.append(Document(
                            page_content=str(content),
                            metadata={
                                "type": elem_type,
                                "original_type": elem_type,
                                "source": elem_type,
                                "file_path": file_path,
                                "session_id": self.session_id,
                                "retriever_type": self.retriever_type,
                                "file": file_name,
                                "page": elem.get('page', 1),
                                "file_type": file_type,
                                "embedding_model": self.embedding_model_name,
                                "chunk_id": f"{file_name}_{elem_type}"
                            }
                        ))

            if documents:
                if self.vector_store is None:
                    try:
                        self.vector_store = FAISS.from_documents(
                            documents=documents,
                            embedding=self.embeddings
                        )
                        self.vector_store.save_local(self.db_dir)
                    except Exception as e:
                        print(f"Error creating vector store: {e}")
                        return [], "", {}
                else:
                    try:
                        self.vector_store.add_documents(documents)
                        self.vector_store.save_local(self.db_dir)
                    except Exception as e:
                        print(f"Error updating vector store: {e}")
                        return [], "", {}

                self.last_used_time = time.time()
                self._files.add(file_path)
                
                # Extract text for summary
                full_text = "\n".join([d.page_content for d in documents])
                return documents, full_text, structured_data
            else:
                return [], "", {}

        except Exception:
            return [], "", {}

    async def add_file(self, file_path: str) -> Dict[str, Any]:
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.txt':
                return {"success": False, "message": "不支持的文本类型"}
            # if file_path in self._files:
            #     return {"success": True, "message": "文件已存在", "file": file_path}
            
            docs, content_text, structured_data = await self.process_and_embed(file_path)
            
            # 收集统计信息
            stats = {
                "success": bool(docs),
                "count": len(docs),
                "file": file_path,
                "elements_count": len(docs),
                "content_preview": content_text[:3000] if content_text else "",
                "structured_data": structured_data
            }
            return stats
        except Exception as e:
            return {"success": False, "error": str(e), "file": file_path}

    def _get_file_type(self, file_path: str) -> str:
        """
        获取文件类型

        Args:
            file_path: 文件路径

        Returns:
            文件类型字符串
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.ptf']:
            return 'ptf'
        if ext in ['.pdf']:
            return 'pdf'
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            return 'image'
        else:
            return 'unknown'

    def search(self, query: str, k: int = 5, file_paths: List[str] = None) -> List[Document]:
        """
        搜索相关文档

        Args:
            query: 搜索查询
            k: 返回结果数量
            file_paths: 指定检索的文件路径列表

        Returns:
            相关文档列表
        """
        try:
            if not query or not str(query).strip():
                return []
            if self.vector_store is None:
                return []

            self.last_used_time = time.time()
            
            filter_dict = None
            if file_paths:
                if len(file_paths) == 1:
                    filter_dict = {"file_path": file_paths[0]}
                else:
                    filter_dict = {"file_path": {"$in": file_paths}}
            
            results = self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            return results

        except Exception:
            return []

    def should_cleanup(self, max_age_hours: float = 24.0) -> bool:
        """
        检查是否应该清理（基于最后使用时间）

        Args:
            max_age_hours: 最大闲置小时数

        Returns:
            是否需要清理
        """
        current_time = time.time()
        age_hours = (current_time - self.last_used_time) / 3600.0
        return age_hours > max_age_hours

    def cleanup(self) -> bool:
        """
        清理会话数据

        Returns:
            是否清理成功
        """
        try:
            if self.vector_store:
                self.vector_store = None

            if os.path.exists(self.db_dir):
                shutil.rmtree(self.db_dir)
                return True

            return False
        except Exception:
            return False

    def get_stats(self) -> dict:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "session_id": self.session_id,
            "retriever_type": self.retriever_type,
            "embedding_model": self.embedding_model_name,
            "db_dir": self.db_dir,
            "created_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.created_time)),
            "last_used_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_used_time)),
            "age_seconds": time.time() - self.created_time,
            "idle_seconds": time.time() - self.last_used_time
        }

        try:
            if self.vector_store:
                count = self.vector_store.index.ntotal
                stats["document_count"] = count
            else:
                stats["document_count"] = 0
        except Exception:
            stats["document_count"] = 0

        stats["vector_store_initialized"] = bool(self.vector_store)
        stats["ocr_available"] = bool(self.ocr_available)

        return stats

    def close(self):
        """关闭资源"""
        pass
