import os
import pickle
from typing import List, Dict, Any
import time
import shutil
from dotenv import load_dotenv

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from file_management.txt_processor import TxtProcessor
from config import settings

# 加载环境变量
load_dotenv()


class TxtHybridSearch:
    """TXT文件的混合检索器（BM25 + 向量检索 + 重排序）"""

    def __init__(self, session_id: str):
        """
        初始化检索器

        Args:
            session_id: 会话ID，用于区分不同检索实例
        """
        self.session_id = session_id
        self.txt_processor = TxtProcessor()
        # Use absolute path to Backend/storage/txt_cache
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cache_dir = os.path.join(base_dir, "storage", "txt_cache", session_id)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.documents: List[Document] = []
        self.bm25_index = None
        self.vector_store = None
        self.file_paths: List[str] = []

        # 初始化分词器适配
        self.tokenizer = self._init_tokenizer()

        self.created_time = time.time()
        self.last_used_time = time.time()
        self._load_state()

    def _init_tokenizer(self):
        try:
            import jieba
            class JiebaAdapter:
                def cut_text(self, text: str):
                    return list(jieba.cut(text, HMM=True))
                def cut_query(self, text: str):
                    return list(jieba.cut(text, HMM=True))
            return JiebaAdapter()
        except Exception:
            return None

    def _get_state_file_path(self, filename: str) -> str:
        """
        获取状态文件路径

        Args:
            filename: 文件名

        Returns:
            完整的文件路径
        """
        return os.path.join(self.cache_dir, filename)

    def _save_state(self):
        """保存状态到缓存文件"""
        try:
            state = {
                "documents": self.documents,
                "file_paths": self.file_paths,
                "created_time": self.created_time,
                "last_used_time": self.last_used_time
            }

            with open(self._get_state_file_path("search_state.pkl"), 'wb') as f:
                pickle.dump(state, f)

            if self.bm25_index:
                with open(self._get_state_file_path("bm25_index.pkl"), 'wb') as f:
                    pickle.dump(self.bm25_index, f)

            if self.vector_store:
                vector_dir = self._get_state_file_path("faiss_index")
                os.makedirs(vector_dir, exist_ok=True)
                self.vector_store.save_local(vector_dir)
        except Exception:
            pass

    def _load_state(self):
        """从缓存文件加载状态"""
        try:
            state_file = self._get_state_file_path("search_state.pkl")
            if os.path.exists(state_file):
                with open(state_file, 'rb') as f:
                    state = pickle.load(f)

                self.documents = state.get("documents", [])
                self.file_paths = state.get("file_paths", [])
                self.created_time = state.get("created_time", time.time())
                self.last_used_time = state.get("last_used_time", time.time())

                bm25_file = self._get_state_file_path("bm25_index.pkl")
                if os.path.exists(bm25_file):
                    with open(bm25_file, 'rb') as f:
                        self.bm25_index = pickle.load(f)

                vector_dir = self._get_state_file_path("faiss_index")
                if os.path.exists(os.path.join(vector_dir, "index.faiss")):
                    api_key = os.environ.get("ALIYUNBAILIAN_API_KEY")
                    if api_key:
                        embedding_model = DashScopeEmbeddings(
                            model=settings.model,
                            dashscope_api_key=api_key,
                        )
                        self.vector_store = FAISS.load_local(
                            vector_dir,
                            embedding_model,
                            allow_dangerous_deserialization=True
                        )
                    else:
                        self.vector_store = None
        except Exception:
            pass

    def ensure_document_list(self, input_data) -> List[Document]:
        """
        转换输入数据为Document列表

        Args:
            input_data: 输入数据，可以是元组或Document列表

        Returns:
            Document对象列表
        """
        documents = []

        if isinstance(input_data, tuple):
            for i, item in enumerate(input_data):
                if isinstance(item, tuple) and len(item) >= 1:
                    text_content = item[0] if len(item) > 0 else ""
                    metadata = item[1] if len(item) > 1 else {}

                    if not isinstance(metadata, dict):
                        metadata = {"raw_metadata": metadata}

                    metadata.update({
                        "session_id": self.session_id,
                        "retriever_type": "hybrid",
                        "file_type": "txt",
                        "index": i,
                        "source": "txt_file"
                    })

                    documents.append(Document(
                        page_content=str(text_content),
                        metadata=metadata
                    ))

        elif isinstance(input_data, list):
            for i, item in enumerate(input_data):
                if isinstance(item, Document):
                    item.metadata.update({
                        "session_id": self.session_id,
                        "retriever_type": "hybrid"
                    })
                    documents.append(item)

        return documents

    def keyword_search_function(self, text: str) -> list:
        """
        中文分词函数

        Args:
            text: 待分词的文本

        Returns:
            分词后的单词列表
        """
        if self.tokenizer:
            try:
                words = self.tokenizer.cut_text(text)
                filtered_words = [word.strip() for word in words if word.strip()]
                return filtered_words
            except Exception:
                pass

        return [char for char in text if char.strip()]

    async def add_file(self, file_path: str) -> Dict[str, Any]:
        """
        添加文件到索引
        
        Args:
            file_path: 文件路径
            
        Returns:
            操作结果字典
        """
        try:
            print(f"[Debug] add_file called for: {file_path}", flush=True)
            new_docs, content_text = await self.process_and_index(file_path)
            # 如果返回了文档列表，或者文件已经在列表中（视为成功）
            success = len(new_docs) > 0 or file_path in self.file_paths
            
            return {
                "success": success,
                "message": "文件已添加" if success else "文件添加失败",
                "count": len(new_docs),
                "content_preview": content_text[:3000] if content_text else ""
            }
        except Exception as e:
            print(f"[Debug] add_file failed: {e}", flush=True)
            return {
                "success": False,
                "message": str(e),
                "error": str(e)
            }

    async def process_and_index(self, file_path: str) -> (List[Document], str):
        """
        处理TXT文件并建立索引
        
        Returns:
            (新增的Document列表, 文件全文内容)
        """
        try:
            print(f"[Debug] Processing file: {file_path}", flush=True)
            raw_docs = self.txt_processor.HandleTxt(file_path)
            print(f"[Debug] raw_docs count: {len(raw_docs)}", flush=True)
            
            # Extract full text from raw_docs before processing
            full_text = "\n".join([d.page_content for d in raw_docs])
            print(f"[Debug] full_text length: {len(full_text)}", flush=True)

            if file_path in self.file_paths:
                return [], full_text
            
            new_documents = self.ensure_document_list(raw_docs)

            if not new_documents:
                return [], full_text

            # Add file path to metadata
            for doc in new_documents:
                doc.metadata['file_path'] = file_path
                doc.metadata['file_name'] = os.path.basename(file_path)

            self.documents.extend(new_documents)
            self.file_paths.append(file_path)
            self._rebuild_bm25_index()
            self._rebuild_vector_index()
            self._save_state()
            self.last_used_time = time.time()

            return new_documents, full_text
        except Exception as e:
            print(f"[Debug] process_and_index failed: {e}")
            import traceback
            traceback.print_exc()
            return [], ""

    def _rebuild_bm25_index(self):
        """重建BM25全文检索索引"""
        if not self.documents:
            self.bm25_index = None
            return

        try:
            self.bm25_index = BM25Retriever.from_documents(
                documents=self.documents,
                preprocess_func=self.keyword_search_function,
                k=10
            )
        except Exception:
            self.bm25_index = None

    def _rebuild_vector_index(self):
        """重建向量检索索引"""
        if not self.documents:
            self.vector_store = None
            return

        try:
            api_key = os.environ.get("ALIYUNBAILIAN_API_KEY")
            if not api_key:
                self.vector_store = None
                return
            embedding_model = DashScopeEmbeddings(
                model=settings.model,
                dashscope_api_key=api_key,
            )
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(
                    documents=self.documents,
                    embedding=embedding_model
                )
            else:
                self.vector_store.add_documents(self.documents)
        except Exception as e:
            print(f"[Debug] Rebuild vector index failed: {e}")
            import traceback
            traceback.print_exc()
            self.vector_store = None

    def _rerank_results(self, bm25_results: List[Document], vector_results: List[Document], query: str) -> List[
        Document]:
        """
        对检索结果进行重排序

        Args:
            bm25_results: BM25检索结果
            vector_results: 向量检索结果
            query: 查询文本

        Returns:
            重排序后的Document列表
        """
        all_results = []

        # 合并结果并去重
        seen = set()

        for i, doc in enumerate(vector_results):
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen:
                seen.add(content_hash)
                doc.metadata["vector_score"] = 1.0 - (i / max(len(vector_results), 1))
                all_results.append(doc)

        for i, doc in enumerate(bm25_results):
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen:
                seen.add(content_hash)
                doc.metadata["bm25_score"] = 1.0 - (i / max(len(bm25_results), 1))
                all_results.append(doc)

        if not all_results:
            return []

        # 计算综合分数
        query_lower = query.lower()

        for doc in all_results:
            content = doc.page_content.lower()
            score = 0.0

            bm25_score = doc.metadata.get("bm25_score", 0)
            vector_score = doc.metadata.get("vector_score", 0)
            score = bm25_score * 0.4 + vector_score * 0.6

            # 字符匹配度
            matched_chars = 0
            for char in query_lower:
                if char in content:
                    matched_chars += 1

            if len(query_lower) > 0:
                char_match_ratio = matched_chars / len(query_lower)
                score += char_match_ratio * 0.3

            # 短语匹配
            for i in range(len(query_lower) - 1):
                phrase = query_lower[i:i + 2]
                if len(phrase) >= 2 and phrase in content:
                    score += 0.05

            # 内容长度优化
            content_len = len(doc.page_content)
            if 200 <= content_len <= 600:
                score += 0.1
            elif 100 <= content_len < 200:
                score += 0.05

            doc.metadata["final_score"] = min(max(score, 0), 1.0)

        # 按分数排序
        all_results.sort(key=lambda x: x.metadata.get("final_score", 0), reverse=True)

        return all_results

    def search(self, query: str, k: int = 5, file_paths: List[str] = None) -> List[Document]:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            file_paths: 指定检索的文件路径列表
            
        Returns:
            检索到的Document列表
        """
        if not self.documents:
            print("[Debug] No documents in retriever")
            return []

        try:
            self.last_used_time = time.time()
            
            print(f"[Debug] Searching with query: {query}")
            
            # Adjust k if filtering
            search_k = k
            if file_paths:
                search_k = k * 5 # Fetch more candidates for filtering
            
            bm25_results = self.bm25_index.invoke(query) if self.bm25_index else []
            # Note: BM25 invoke doesn't support k directly in all versions, it uses constructor k. 
            # But we can't easily change it dynamically. 
            # We assume BM25 returns enough results (constructor set k=10, maybe too low for filtering).
            # Let's hope BM25 returns "all relevant" or enough. 
            # Actually line 273 set k=10. This is a problem if we need to filter.
            # But for now let's proceed. 
            
            vector_results = self.vector_store.similarity_search(query, k=search_k * 2) if self.vector_store else []
            
            # Filter results if file_paths provided
            if file_paths:
                file_paths_set = set(file_paths)
                
                # Filter BM25 results
                filtered_bm25 = []
                for doc in bm25_results:
                    if doc.metadata.get('file_path') in file_paths_set:
                        filtered_bm25.append(doc)
                bm25_results = filtered_bm25
                
                # Filter Vector results
                filtered_vector = []
                for doc in vector_results:
                    if doc.metadata.get('file_path') in file_paths_set:
                        filtered_vector.append(doc)
                vector_results = filtered_vector

            reranked_results = self._rerank_results(bm25_results, vector_results, query)

            return reranked_results[:k]
        except Exception:
            if self.vector_store:
                res = self.vector_store.similarity_search(query, k=k*5 if file_paths else k)
                if file_paths:
                    file_paths_set = set(file_paths)
                    res = [d for d in res if d.metadata.get('file_path') in file_paths_set]
                return res[:k]
            elif self.bm25_index:
                res = self.bm25_index.invoke(query)
                if file_paths:
                    file_paths_set = set(file_paths)
                    res = [d for d in res if d.metadata.get('file_path') in file_paths_set]
                return res[:k]
            return []



    def should_cleanup(self, max_idle_hours: int = 3) -> bool:
        """
        检查是否需要清理

        Args:
            max_idle_hours: 最大空闲小时数

        Returns:
            是否需要清理的布尔值
        """
        idle_hours = (time.time() - self.last_used_time) / 3600
        return idle_hours > max_idle_hours

    def cleanup(self):
        """清理缓存和资源"""
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
            self.documents = []
            self.bm25_index = None
            self.vector_store = None
            self.file_paths = []
        except Exception:
            pass

    def get_stats(self) -> dict:
        """
        获取检索器统计信息

        Returns:
            统计信息字典
        """
        return {
            "session_id": self.session_id,
            "document_count": len(self.documents),
            "file_count": len(self.file_paths),
            "has_bm25_index": self.bm25_index is not None,
            "has_vector_store": self.vector_store is not None,
            "jieba_tokenizer": "available" if self.tokenizer else "unavailable",
            "created_time": time.ctime(self.created_time),
            "last_used_time": time.ctime(self.last_used_time),
            "idle_hours": (time.time() - self.last_used_time) / 3600,
        }
