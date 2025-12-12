import os
import time
import asyncio
from typing import List
from langchain_core.documents import Document
from search.txt_hybrid_search import TxtHybridSearch
from search.ptf_search import ptf_search

class HybridRetriever:
    def __init__(self, txt_retriever: TxtHybridSearch, ptf_retriever: ptf_search):
        self.txt_retriever = txt_retriever
        self.ptf_retriever = ptf_retriever
        self.session_id = "hybrid_" + str(time.time())
        self.tokenizer = self._init_tokenizer()

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

    def _simple_rerank(self, results: List[Document], query: str) -> List[Document]:
        if not results:
            return []
        query_lower = str(query).lower()
        if self.tokenizer:
            try:
                query_words = self.tokenizer.cut_query(query_lower)
            except Exception:
                query_words = [c for c in query_lower if c.strip()]
        else:
            query_words = [c for c in query_lower if c.strip()]

        for doc in results:
            content = doc.page_content
            metadata = getattr(doc, 'metadata', {}) or {}
            content_type = metadata.get('type', 'text')
            score = 0.0
            if content_type == 'text':
                score += 0.4
            elif content_type == 'table':
                score += 0.35
            elif content_type == 'figure':
                score += 0.3
            elif content_type == 'formula':
                score += 0.25
            elif content_type == 'image':
                score += 0.2

            content_lower = str(content).lower()
            matched_words = 0
            for word in query_words:
                w = str(word).strip()
                if len(w) > 1 and w in content_lower:
                    matched_words += 1
                    score += 0.03 if len(w) >= 3 else 0.01

            if query_words:
                match_ratio = matched_words / max(len(query_words), 1)
                score += match_ratio * 0.6

            if query_lower in content_lower:
                score += 0.15

            first_200 = content_lower[:200]
            for word in query_words:
                w = str(word).strip()
                if len(w) > 1 and w in first_200:
                    score += 0.05

            doc.metadata['rerank_score'] = min(max(score, 0), 1.0)
            doc.metadata['matched_words'] = matched_words
            doc.metadata['total_words'] = len(query_words)

        results.sort(key=lambda x: x.metadata.get('rerank_score', 0), reverse=True)

        final_results = []
        type_counts = {}

        for doc in results:
            t = doc.metadata.get('type', 'text')
            cnt = type_counts.get(t, 0)
            if cnt < 3:
                final_results.append(doc)
                type_counts[t] = cnt + 1
            if len(final_results) >= 10:
                break

        if len(final_results) < min(len(results), 8):
            for doc in results:
                if doc not in final_results:
                    final_results.append(doc)
                if len(final_results) >= min(len(results), 8):
                    break

        for i, doc in enumerate(final_results):
            doc.metadata['rank'] = i + 1

        return final_results

    def search(self, query: str, k: int = 5, file_paths: List[str] = None) -> List[Document]:
        all_results = []
        
        # Split file paths if provided
        txt_files = None
        ptf_files = None
        
        if file_paths is not None:
            txt_files = []
            ptf_files = []
            for fp in file_paths:
                ext = os.path.splitext(fp)[1].lower()
                if ext in ['.txt', '.md']:
                    txt_files.append(fp)
                else:
                    ptf_files.append(fp)
        
        # Search TXT if no paths specified or txt paths exist
        if file_paths is None or txt_files:
            try:
                txt_results = self.txt_retriever.search(query, k=k * 2, file_paths=txt_files)
                for doc in txt_results:
                    if not hasattr(doc, 'metadata') or doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['source'] = 'txt'
                    doc.metadata['type'] = doc.metadata.get('type', 'text')
                all_results.extend(txt_results)
            except Exception:
                pass

        # Search PTF if no paths specified or ptf paths exist
        if file_paths is None or ptf_files:
            try:
                ptf_results = self.ptf_retriever.search(query, k=k * 2, file_paths=ptf_files)
                for doc in ptf_results:
                    if not hasattr(doc, 'metadata') or doc.metadata is None:
                        doc.metadata = {}
                    if 'type' not in doc.metadata:
                        doc.metadata['type'] = 'text'
                    doc.metadata['source'] = 'ptf'
                all_results.extend(ptf_results)
            except Exception:
                pass

        reranked = self._simple_rerank(all_results, query)
        return reranked[:k]

    async def add_file(self, file_path: str) -> dict:
        try:
            print(f"[Debug] HybridRetriever.add_file: {file_path}", flush=True)
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.txt', '.md']:
                return await self.txt_retriever.add_file(file_path)
            else:
                print(f"[Debug] Delegating to ptf_retriever...", flush=True)
                res = await self.ptf_retriever.add_file(file_path)
                print(f"[Debug] ptf_retriever returned: {str(res)[:100]}", flush=True)
                return res
        except Exception as e:
            print(f"[Debug] HybridRetriever.add_file failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return {"success": False, "message": str(e), "error": str(e)}

    def get_stats(self):
        stats = {"session_id": self.session_id, "retriever_type": "hybrid", "components": ["txt_retriever", "ptf_retriever"]}
        try:
            if hasattr(self.txt_retriever, 'get_stats'):
                stats['txt_stats'] = self.txt_retriever.get_stats()
        except Exception:
            pass
        try:
            if hasattr(self.ptf_retriever, 'get_stats'):
                stats['ptf_stats'] = self.ptf_retriever.get_stats()
        except Exception:
            pass
        return stats

    def cleanup(self):
        try:
            if hasattr(self.txt_retriever, 'cleanup'):
                self.txt_retriever.cleanup()
        except Exception:
            pass
        try:
            if hasattr(self.ptf_retriever, 'cleanup'):
                self.ptf_retriever.cleanup()
        except Exception:
            pass


def get_hybrid_retriever(session_id: str, file_paths: List[str] = None) -> HybridRetriever:
    txt_retriever = TxtHybridSearch(session_id)
    ptf_retriever = ptf_search(session_id)
    if file_paths:
        for file_path in file_paths:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                loop = asyncio.get_event_loop()
                if ext == '.txt':
                    if loop.is_running():
                        with asyncio.Runner() as runner:
                            runner.run(txt_retriever.process_and_index(file_path))
                    else:
                        asyncio.run(txt_retriever.process_and_index(file_path))
                else:
                    if loop.is_running():
                        with asyncio.Runner() as runner:
                            runner.run(ptf_retriever.process_and_embed(file_path))
                    else:
                        asyncio.run(ptf_retriever.process_and_embed(file_path))
            except Exception:
                continue
    return HybridRetriever(txt_retriever, ptf_retriever)
