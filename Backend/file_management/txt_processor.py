from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  # 导入Document类
from typing import List

class TxtProcessor:
    """
    文本文件处理器，用于读取和分割文本文件。
    """
    def HandleTxt(self, txt_path: str) -> List[Document]:
        """
        读取并分割文本文件。

        使用 RecursiveCharacterTextSplitter 对文本进行分块处理，
        以便于后续的 RAG 检索或 LLM 处理。

        Args:
            txt_path (str): 文本文件的绝对路径。

        Returns:
            List[Document]: 分割后生成的 LangChain Document 对象列表。
        """
        # 1. 加载文本分割器
        text_splitter_optimized = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            length_function=len
        )

        # 2. 读取文件内容
        # 确保使用正确的编码打开，并读取为字符串
        with open(txt_path, "r", encoding="utf-8") as f:
            text_content = f.read()  # 关键修正：使用 .read() 获取字符串

        # 3. 创建初始文档对象
        # RecursiveCharacterTextSplitter.split_documents 需要接受 Document 对象列表
        initial_document = [Document(page_content=text_content)]

        # 4. 分割文档
        # split_documents 方法专用于分割Document对象
        all_splits = text_splitter_optimized.split_documents(initial_document)

        return all_splits
