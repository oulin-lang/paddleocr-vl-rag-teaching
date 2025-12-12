import os
import json
import base64
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from datetime import datetime

# 加载环境变量
load_dotenv()


from runtime_config import runtime_config

class ContentType(Enum):
    """文档内容类型枚举"""
    TEXT = "text"  # 文本
    TITLE = "title"  # 标题
    TABLE = "table"  # 表格
    FORMULA = "formula"  # 公式
    FIGURE = "figure"  # 图表/图片
    LIST = "list"  # 列表
    HEADING = "heading"  # 标题
    CAPTION = "caption"  # 图注


@dataclass
class DocumentElement:
    """文档元素数据结构

    存储解析出的文档元素信息
    """
    element_id: str  # 元素唯一标识
    content_type: ContentType  # 内容类型
    content: Any  # 元素内容
    bbox: Tuple[float, float, float, float]  # 边界框坐标
    page_num: int  # 页码
    confidence: float = 1.0  # 置信度
    metadata: Dict[str, Any] = None  # 额外元数据

    def __post_init__(self):
        """初始化后处理，确保metadata不为None"""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            字典格式的元素数据
        """
        return {
            "id": self.element_id,
            "type": self.content_type.value,
            "content": self.content,
            "bbox": self.bbox,
            "page": self.page_num,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


from .office_processor import OfficeProcessor

class PaddlePTFParser:
    """多模态文档解析器

    使用PaddleOCR-VL API解析PDF、图片等文档
    """

    def __init__(self, session_id: str, output_dir: str = None):
        """初始化解析器

        Args:
            session_id: 会话ID
            output_dir: 输出目录路径 (默认为 Backend/storage/output)
        """
        self.session_id = session_id
        self.office_processor = OfficeProcessor()

        # 从环境变量或默认值获取API配置
        self.api_url = os.getenv("PADDLEOCR_VL_API_URL")
        self.api_key = os.getenv("PADDLEOCR_VL_TOKEN")

        if not self.api_url or not self.api_key:
            raise ValueError("API配置缺失")

        # 创建输出目录
        if output_dir is None:
            # 默认指向 Backend/storage/output
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, "storage", "output")
            
        self.output_dir = Path(output_dir) / session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 存储解析结果
        self.elements: List[DocumentElement] = []
        self.processed_files: List[str] = []
        self.start_time = time.time()

    def parse_file(self, file_path: str, enable_chart_recognition: bool = True) -> Dict[str, Any]:
        """解析文档文件
        
        Args:
            file_path: 文件路径
            enable_chart_recognition: 是否启用图表识别
            
        Returns:
            解析结果字典
        """
        print(f"[Debug] PaddlePTFParser.parse_file: {file_path}", flush=True)
        if not os.path.exists(file_path):
            return {"error": f"文件不存在: {file_path}"}

        file_type = self._detect_file_type(file_path)
        print(f"[Debug] Detected file type: {file_type}", flush=True)
        if file_type == 'unknown':
            return {"error": f"不支持的文件格式: {file_path}"}
            
        if file_type == 'office':
            return self.office_processor.parse(file_path)

        try:
            # 1. 编码文件为base64
            print(f"[Debug] Encoding file to base64...", flush=True)
            file_base64 = self._encode_file_to_base64(file_path)
            print(f"[Debug] Encoded size: {len(file_base64)}", flush=True)

            # 2. 调用布局解析API
            print(f"[Debug] Calling layout parsing API...", flush=True)
            api_result = self._call_layout_parsing_api(file_base64, file_type, enable_chart_recognition)
            print(f"[Debug] API call finished. Result keys: {api_result.keys() if api_result else 'None'}", flush=True)

            if not api_result or "result" not in api_result:
                error_msg = api_result.get("error", "API返回空结果") if isinstance(api_result, dict) else "API调用失败"
                return {"error": error_msg}

            result_data = api_result["result"]
            print(f"[Debug] Processing result_data...", flush=True)

            # 3. 解析结构化数据
            all_elements = []
            extracted_texts = []
            extracted_images = []
            
            layout_results = result_data.get("layoutParsingResults", [])
            print(f"[Debug] Found {len(layout_results)} layout results", flush=True)
            
            for i, page_result in enumerate(layout_results):
                page_num = i + 1
                # 获取Markdown文本
                if "markdown" in page_result and "text" in page_result["markdown"]:
                    md_text = page_result["markdown"]["text"]
                    extracted_texts.append(md_text)
                    self._save_page_markdown(md_text, page_num)
                
                # 解析页面元素
                page_elements = self._parse_page_result(page_result, page_num)
                all_elements.extend(page_elements)

                # 提取图片
                page_images = self._extract_page_images(page_result, page_num)
                extracted_images.extend(page_images)

            print(f"[Debug] Finished processing layout results", flush=True)
            
            # 4. 保存到当前对象
            self.elements.extend(all_elements)
            self.processed_files.append(file_path)

            # 5. 生成结构化数据
            structured_data = self._generate_structured_data(all_elements, file_path, extracted_images)

            # 6. 保存所有结果
            output_files = self._save_all_results(structured_data, file_path, all_elements)
            
            # 构造最终返回结果
            final_result = {
                "elements": [e.to_dict() for e in all_elements],
                "structured_data": structured_data,
                "full_text": "\n\n".join(extracted_texts),
                "file_type": file_type,
                "file_path": file_path,
                "meta": {
                    "total_pages": len(layout_results),
                    "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "output_files": output_files
            }
            print(f"[Debug] Returning final result", flush=True)
            return final_result

        except Exception as e:
            print(f"解析文件失败: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def _detect_file_type(self, file_path: str) -> str:
        """检测文件类型

        Args:
            file_path: 文件路径

        Returns:
            文件类型字符串
        """
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            return 'pdf'
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']:
            return 'image'
        elif ext in ['.docx', '.pptx']:
            return 'office'
        return 'unknown'

    def _encode_file_to_base64(self, file_path: str) -> str:
        """将文件编码为base64格式

        Args:
            file_path: 文件路径

        Returns:
            base64编码的字符串
        """
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        return base64.b64encode(file_bytes).decode("ascii")

    def _call_layout_parsing_api(self, file_base64: str, file_type: str, enable_chart: bool = True) -> Dict:
        """调用布局解析API

        Args:
            file_base64: base64编码的文件内容
            file_type: 文件类型
            enable_chart: 是否启用图表识别

        Returns:
            API响应结果
        """
        # 检查联网权限
        if runtime_config and not runtime_config.enable_mcp_access:
            return {"error": "Network functions disabled (OCR)"}

        file_type_num = 0 if file_type == 'pdf' else 1

        payload = {
            "file": file_base64,
            "fileType": file_type_num,
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": enable_chart,
            "useLayoutDetection": True,
            "useTableStructure": True,
        }

        headers = {
            "Authorization": f"token {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            print("[Debug] Calling layout parsing API...", flush=True)
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                print(f"[Error] API returned status code {response.status_code}: {response.text}", flush=True)
                return {"error": f"API request failed with status {response.status_code}"}
                
            result = response.json()
            print("[Debug] API call successful", flush=True)
            return result
        except requests.exceptions.Timeout:
            print("[Error] API request timed out", flush=True)
            return {"error": "API request timed out"}
        except Exception as e:
            print(f"[Error] API request failed: {e}", flush=True)
            return {"error": f"API request failed: {str(e)}"}

    def _parse_page_result(self, page_result: Dict, page_num: int) -> List[DocumentElement]:
        """解析单页API返回结果

        Args:
            page_result: 单页API返回数据
            page_num: 页码

        Returns:
            文档元素列表
        """
        elements = []

        # 提取Markdown文本
        markdown_text = page_result.get("markdown", {}).get("text", "")
        if markdown_text:
            elements.append(DocumentElement(
                element_id=f"page_{page_num}_markdown",
                content_type=ContentType.TEXT,
                content=markdown_text,
                bbox=(0, 0, 1000, 1000),
                page_num=page_num,
                metadata={"source": "markdown", "format": "full_page"}
            ))

        # 解析布局检测结果
        if "layout_det_res" in page_result:
            layout_data = page_result["layout_det_res"]
            if isinstance(layout_data, dict) and "boxes" in layout_data:
                for i, box in enumerate(layout_data["boxes"]):
                    if isinstance(box, dict):
                        label = box.get("label", "unknown")
                        content = box.get("content", "")
                        bbox = box.get("coordinate", [0, 0, 0, 0])
                        confidence = box.get("score", 1.0)

                        content_type = ContentType.TEXT
                        if label in ["table", "表格"]:
                            content_type = ContentType.TABLE
                        elif label in ["formula", "公式", "math"]:
                            content_type = ContentType.FORMULA
                        elif label in ["figure", "图片", "chart", "图表", "image"]:
                            content_type = ContentType.FIGURE
                        elif label in ["title", "标题", "heading"]:
                            content_type = ContentType.TITLE
                        elif label in ["list", "列表"]:
                            content_type = ContentType.LIST

                        elements.append(DocumentElement(
                            element_id=f"page_{page_num}_layout_{i}",
                            content_type=content_type,
                            content=content,
                            bbox=tuple(bbox),
                            page_num=page_num,
                            confidence=confidence,
                            metadata={
                                "source": "layout_detection",
                                "label": label,
                                "original_index": i
                            }
                        ))

        # 解析解析结果列表
        if "parsing_res_list" in page_result:
            parsing_list = page_result["parsing_res_list"]
            for i, item in enumerate(parsing_list):
                if isinstance(item, dict):
                    label = item.get("block_label", item.get("type", "text"))
                    content = item.get("block_content", item.get("content", ""))
                    bbox = item.get("block_bbox", item.get("bbox", [0, 0, 0, 0]))

                    content_type = ContentType.TEXT
                    if "table" in label.lower():
                        content_type = ContentType.TABLE
                    elif "formula" in label.lower() or "math" in label.lower():
                        content_type = ContentType.FORMULA
                    elif any(img_key in label.lower() for img_key in ["figure", "image", "chart", "diagram"]):
                        content_type = ContentType.FIGURE

                    elements.append(DocumentElement(
                        element_id=f"page_{page_num}_parsing_{i}",
                        content_type=content_type,
                        content=content,
                        bbox=tuple(bbox),
                        page_num=page_num,
                        metadata={"source": "parsing_res_list", "label": label}
                    ))

        return elements

    def _save_page_markdown(self, md_content: str, page_num: int):
        """保存页面Markdown内容到文件

        Args:
            md_content: Markdown内容
            page_num: 页码
        """
        if not self.output_dir.exists():
             try:
                 self.output_dir.mkdir(parents=True, exist_ok=True)
             except Exception as e:
                 print(f"[Error] Failed to create output directory {self.output_dir}: {e}")

        md_filename = self.output_dir / f"page_{page_num:03d}.md"
        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(md_content)

    def _extract_page_images(self, page_result: Dict, page_num: int) -> List[Dict]:
        """提取页面中的图片

        Args:
            page_result: 单页API返回数据
            page_num: 页码

        Returns:
            图片信息列表
        """
        extracted_images = []

        # 检查联网权限
        if runtime_config and not runtime_config.enable_mcp_access:
            return extracted_images

        # 从markdown.images提取
        if "markdown" in page_result and "images" in page_result["markdown"]:
            images_dict = page_result["markdown"]["images"]
            for img_path, img_url in images_dict.items():
                try:
                    img_response = requests.get(img_url, timeout=30)
                    if img_response.status_code == 200:
                        full_img_path = self.output_dir / "images" / f"page_{page_num}" / img_path
                        full_img_path.parent.mkdir(parents=True, exist_ok=True)

                        with open(full_img_path, "wb") as f:
                            f.write(img_response.content)

                        extracted_images.append({
                            "page": page_num,
                            "path": str(full_img_path),
                            "url": img_url,
                            "source": "markdown_images",
                            "filename": img_path
                        })
                except Exception:
                    pass

        # 从outputImages提取
        if "outputImages" in page_result:
            output_images = page_result["outputImages"]
            for img_name, img_url in output_images.items():
                try:
                    img_response = requests.get(img_url, timeout=30)
                    if img_response.status_code == 200:
                        filename = self.output_dir / "images" / f"page_{page_num}_{img_name}.jpg"
                        filename.parent.mkdir(parents=True, exist_ok=True)

                        with open(filename, "wb") as f:
                            f.write(img_response.content)

                        extracted_images.append({
                            "page": page_num,
                            "path": str(filename),
                            "url": img_url,
                            "source": "outputImages",
                            "filename": img_name
                        })
                except Exception:
                    pass

        return extracted_images

    def _generate_structured_data(self, elements: List[DocumentElement], file_path: str, images: List[Dict]) -> Dict:
        """生成结构化数据

        Args:
            elements: 文档元素列表
            file_path: 文件路径
            images: 图片信息列表

        Returns:
            结构化数据字典
        """
        # 按类型分组
        by_type = {}
        for elem in elements:
            elem_type = elem.content_type.value
            if elem_type not in by_type:
                by_type[elem_type] = []
            by_type[elem_type].append(elem.to_dict())

        # 按页面分组
        by_page = {}
        for elem in elements:
            page = elem.page_num
            if page not in by_page:
                by_page[page] = []
            by_page[page].append(elem.to_dict())

        orig_images = [
            img for img in images
            if (img.get("source") == "markdown_images" or not any(s in str(img.get("filename", "")).lower() for s in ["layout", "det", "order", "bbox", "mask", "annot"]))
        ]

        return {
            "session_id": self.session_id,
            "file_name": Path(file_path).name,
            "file_type": self._detect_file_type(file_path),
            "parse_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_pages": max(by_page.keys()) if by_page else 0,
            "total_elements": len(elements),
            "total_images": len(orig_images),
            "elements_by_type": by_type,
            "elements_by_page": by_page,
            "images_summary": [
                {
                    "page": img["page"],
                    "filename": img["filename"],
                    "path": img["path"],
                    "source": img["source"]
                }
                for img in orig_images
            ],
            "processing_time": round(time.time() - self.start_time, 2)
        }

    def _save_all_results(self, structured_data: Dict, file_path: str, elements: List[DocumentElement]) -> Dict:
        """保存所有解析结果到文件

        Args:
            structured_data: 结构化数据
            file_path: 源文件路径
            elements: 文档元素列表

        Returns:
            保存的文件路径信息
        """
        base_name = Path(file_path).stem
        timestamp = int(time.time())

        # 保存结构化JSON
        json_filename = f"{base_name}_{timestamp}_structured.json"
        json_path = self.output_dir / json_filename
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)

        # 生成并保存汇总Markdown文档
        md_filename = f"{base_name}_{timestamp}_report.md"
        md_path = self.output_dir / md_filename
        md_content = self._generate_report_markdown(structured_data)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        # 保存原始元素数据（用于后续向量化）
        elements_filename = f"{base_name}_{timestamp}_elements.json"
        elements_path = self.output_dir / elements_filename
        elements_data = [elem.to_dict() for elem in elements]
        with open(elements_path, 'w', encoding='utf-8') as f:
            json.dump(elements_data, f, ensure_ascii=False, indent=2)

        return {
            "markdown_report": str(md_path),
            "structured_json": str(json_path),
            "elements_json": str(elements_path)
        }

    def _generate_report_markdown(self, data: Dict) -> str:
        """生成汇总报告Markdown内容

        Args:
            data: 结构化数据

        Returns:
            Markdown格式的报告内容
        """
        lines = []
        lines.append(f"# 文档解析汇总报告 - {data['file_name']}\n")
        lines.append(f"**会话ID**: `{data['session_id']}`  ")
        lines.append(f"**文件类型**: {data['file_type']}  ")
        lines.append(f"**解析时间**: {data['parse_time']}  ")
        lines.append(f"**处理耗时**: {data['processing_time']}秒  ")
        lines.append(f"**总页数**: {data['total_pages']}  ")
        lines.append(f"**元素总数**: {data['total_elements']}  ")
        lines.append(f"**图片总数**: {data['total_images']}\n")

        # 元素类型统计
        lines.append("## 元素类型统计\n")
        for elem_type, items in data['elements_by_type'].items():
            lines.append(f"- **{elem_type}**: {len(items)}个")

        # 页面摘要
        lines.append("\n## 页面摘要\n")
        for page_num, elements in data['elements_by_page'].items():
            lines.append(f"### 第 {page_num} 页\n")
            lines.append(f"- **元素数量**: {len(elements)}个")

            page_types = {}
            for elem in elements:
                elem_type = elem['type']
                page_types[elem_type] = page_types.get(elem_type, 0) + 1

            if page_types:
                lines.append("- **元素分布**: " + ", ".join([f"{k}{v}个" for k, v in page_types.items()]))

            lines.append("")

        # 图片信息
        if data['images_summary']:
            lines.append("\n## 提取的图片\n")
            for img in data['images_summary']:
                lines.append(f"- **第{img['page']}页 - {img['filename']}**")
                lines.append(f"  路径: `{img['path']}`  ")
                lines.append(f"  来源: {img['source']}\n")

        # 文件说明
        lines.append("\n---\n")
        lines.append("## 生成的文件说明\n")
        lines.append("1. **页面Markdown文件** (`page_XXX.md`): 每页的完整Markdown内容\n")
        lines.append("2. **汇总报告** (`*_report.md`): 本文件，包含统计和摘要\n")
        lines.append("3. **结构化数据** (`*_structured.json`): 完整的结构化数据\n")
        lines.append("4. **元素数据** (`*_elements.json`): 扁平化元素列表，用于向量化\n")
        lines.append("5. **图片文件夹** (`images/`): 所有提取的图片文件\n")
        lines.append(f"\n*所有文件保存在: `{self.output_dir}`*")

        return "\n".join(lines)
