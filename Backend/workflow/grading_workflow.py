import os
import uuid
import json
import csv
import time
from typing import List, Dict, Any, TypedDict
from pathlib import Path
import asyncio

# ReportLab imports
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# LangGraph imports
from langgraph.graph import StateGraph, END

# Project imports
try:
    from file_management.ptf_processor import PaddlePTFParser
    from agent import Agent
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from file_management.ptf_processor import PaddlePTFParser
    from agent import Agent

from langchain_core.prompts import ChatPromptTemplate

# Register Chinese Font
try:
    pdfmetrics.registerFont(TTFont('SimHei', 'C:\\Windows\\Fonts\\simhei.ttf'))
    FONT_NAME = 'SimHei'
except Exception as e:
    print(f"Warning: Could not register SimHei font: {e}")
    FONT_NAME = 'Helvetica' # Fallback, no Chinese support

class GradingState(TypedDict):
    """
    批改工作流的状态定义。
    """
    files: List[str]
    results: List[Dict[str, Any]]
    output_dir: str
    session_id: str
    summary_path: str

class GradingWorkflow:
    """
    批量作业批改工作流。
    使用 LangGraph 管理批改流程，ReportLab 生成 PDF 报告。
    """
    def __init__(self, output_dir: str = None):
        """
        初始化 GradingWorkflow。

        Args:
            output_dir (str, optional): 输出目录路径。如果未提供，默认使用 storage/output。
        """
        if output_dir is None:
            # 默认输出路径指向 Backend/storage/output
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "output")
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.agent = Agent()
        self.ptf_parser = None # Initialized in process

        # Initialize Graph
        self.workflow = StateGraph(GradingState)
        
        self.workflow.add_node("process_files", self.process_files_node)
        # self.workflow.add_node("generate_summary", self.generate_summary_node) # Removed summary generation
        
        self.workflow.set_entry_point("process_files")
        self.workflow.add_edge("process_files", END)
        # self.workflow.add_edge("process_files", "generate_summary")
        # self.workflow.add_edge("generate_summary", END)
        
        self.app = self.workflow.compile()

    async def process_files(self, file_paths: List[str]) -> List[str]:
        """
        处理一批文件进行批改。

        Args:
            file_paths (List[str]): 待批改文件的路径列表。

        Returns:
            List[str]: 生成的 PDF 报告文件路径列表。
        """
        session_id = str(uuid.uuid4())
        # Initialize parser here to ensure session_id is fresh
        self.ptf_parser = PaddlePTFParser(session_id=session_id, output_dir=self.output_dir)
        
        initial_state = GradingState(
            files=file_paths,
            results=[],
            output_dir=self.output_dir,
            session_id=session_id,
            summary_path=""
        )
        
        final_state = await self.app.ainvoke(initial_state)
        
        # Collect all generated files
        generated_files = [r['output_file'] for r in final_state['results'] if 'output_file' in r]
        # if final_state['summary_path']:
        #     generated_files.append(final_state['summary_path'])
            
        return generated_files

    async def _process_single_file(self, file_path: str, output_dir: str) -> Dict[str, Any] | None:
        """
        处理单个文件的批改逻辑。

        1. 解析文件内容 (PDF/Image/TXT)。
        2. 调用 Agent 进行批改。
        3. 解析 Agent 返回的 JSON 结果。
        4. 生成 PDF 报告。

        Args:
            file_path (str): 文件路径。
            output_dir (str): 输出目录。

        Returns:
            Dict[str, Any] | None: 批改结果字典，如果处理失败返回 None。
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        content = ""
        
        grading_system_prompt = (
            "你是一个专业批改作业的专家。请根据提供的作业内容（包括题目和学生答案），"
            "进行详细的批改。你需要提供以下信息：\n"
            "1. 原题目\n"
            "2. 学生答案\n"
            "3. 标准答案（如果题目中未提供，请根据你的知识生成）\n"
            "4. 知识点识别\n"
            "5. 解题思路\n"
            "6. 错误分析\n"
            "7. 分数（满分100分，根据完成情况打分）\n\n"
            "请以严格的JSON格式返回结果，不要包含Markdown代码块标记（如```json），"
            "JSON结构如下：\n"
            "{{\n"
            "  \"original_question\": \"...\",\n"
            "  \"student_answer\": \"...\",\n"
            "  \"standard_answer\": \"...\",\n"
            "  \"knowledge_points\": \"...\",\n"
            "  \"solution_approach\": \"...\",\n"
            "  \"error_analysis\": \"...\",\n"
            "  \"score\": 0\n"
            "}}"
        )

        try:
            # 1. Parse File
            if file_ext in ['.pdf', '.jpg', '.jpeg', '.png', '.bmp']:
                print(f"Parsing {file_name}...", flush=True)
                # Run CPU-bound parsing in a thread pool to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                parse_result = await loop.run_in_executor(None, self.ptf_parser.parse_file, file_path)
                
                if "error" in parse_result:
                    print(f"Error parsing {file_name}: {parse_result['error']}", flush=True)
                    return None
                content = parse_result.get("full_text", "")
                print(f"Parsing {file_name} completed. Content length: {len(content)}", flush=True)
            elif file_ext == '.txt':
                print(f"Reading {file_name} directly...", flush=True)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                print(f"Unsupported file type: {file_name}", flush=True)
                return None
            
            if not content.strip():
                print(f"No content extracted from {file_name}", flush=True)
                return None

            # 2. Grade with Agent
            print(f"Grading {file_name}...", flush=True)
            user_input = f"请批改以下作业内容：\n\n{content}"
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", grading_system_prompt),
                ("human", "{input}")
            ])
            
            # Using Agent's LLM directly for structured output
            chain = prompt | self.agent.llm
            print(f"Invoking agent for {file_name}...", flush=True)
            response_msg = await chain.ainvoke({"input": user_input})
            response = response_msg.content
            print(f"Agent finished for {file_name}", flush=True)
            
            # 3. Parse JSON Response
            grading_result = self._parse_json_response(response, file_name)
            
            # 4. Generate PDF Report
            pdf_filename = f"{os.path.splitext(file_name)[0]}_{int(time.time())}.pdf"
            pdf_path = os.path.join(output_dir, pdf_filename)
            self._generate_pdf_report(pdf_path, file_name, grading_result)
            
            # Store result
            grading_result['file_name'] = file_name
            grading_result['output_file'] = pdf_path
            return grading_result
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def process_files_node(self, state: GradingState) -> GradingState:
        """
        LangGraph 节点：并发处理所有文件。

        Args:
            state (GradingState): 当前工作流状态。

        Returns:
            GradingState: 更新后的工作流状态，包含批改结果。
        """
        files = state['files']
        output_dir = state['output_dir']
        
        # Concurrently process all files
        tasks = [self._process_single_file(f, output_dir) for f in files]
        results_maybe = await asyncio.gather(*tasks)
        
        # Filter out None results
        results = [r for r in results_maybe if r is not None]
        
        state['results'] = results
        return state

    def generate_summary_node(self, state: GradingState) -> GradingState:
        """
        LangGraph 节点：生成汇总 CSV 报告（已废弃）。

        Args:
            state (GradingState): 当前工作流状态。

        Returns:
            GradingState: 更新后的工作流状态。
        """
        results = state['results']
        if not results:
            return state
            
        csv_filename = f"summary_{int(time.time())}.csv"
        csv_path = os.path.join(state['output_dir'], csv_filename)
        
        summary_data = []
        for res in results:
            summary_data.append({
                "File": res.get('file_name', 'Unknown'),
                "Score": res.get('score', 0),
                "Knowledge Points": res.get('knowledge_points', ''),
                "Error Analysis": str(res.get('error_analysis', ''))[:100],
                "Report Path": res.get('output_file', '')
            })
            
        if summary_data:
            keys = summary_data[0].keys()
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(summary_data)
            state['summary_path'] = csv_path
            
        return state

    def _parse_json_response(self, response: str, file_name: str) -> Dict[str, Any]:
        """
        解析 LLM 返回的 JSON 格式响应。

        Args:
            response (str): LLM 返回的原始字符串。
            file_name (str): 对应的文件名（用于日志记录）。

        Returns:
            Dict[str, Any]: 解析后的字典。如果解析失败，返回包含错误信息的字典。
        """
        try:
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            elif clean_response.startswith("```"):
                clean_response = clean_response[3:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            return json.loads(clean_response)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON for {file_name}")
            return {
                "original_question": "Parse Error",
                "student_answer": "Parse Error",
                "standard_answer": "Parse Error",
                "knowledge_points": "Parse Error",
                "solution_approach": "Parse Error",
                "error_analysis": response,
                "score": 0
            }

    def _generate_pdf_report(self, path: str, original_filename: str, data: Dict[str, Any]):
        """
        使用 ReportLab 生成 PDF 批改报告。

        Args:
            path (str): PDF 文件输出路径。
            original_filename (str): 原始作业文件名。
            data (Dict[str, Any]): 批改结果数据。
        """
        doc = SimpleDocTemplate(path, pagesize=A4)
        styles = getSampleStyleSheet()
        
        # Custom style for Chinese
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontName=FONT_NAME,
            fontSize=10,
            leading=14,
            spaceAfter=6
        )
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName=FONT_NAME,
            fontSize=16,
            leading=20,
            spaceAfter=12,
            alignment=1 # Center
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName=FONT_NAME,
            fontSize=12,
            leading=16,
            spaceAfter=10,
            textColor=colors.darkblue
        )

        elements = []
        
        # Title
        elements.append(Paragraph(f"作业批改报告: {original_filename}", title_style))
        elements.append(Spacer(1, 12))
        
        # Score
        score = data.get('score', 0)
        score_color = colors.green if score >= 60 else colors.red
        score_style = ParagraphStyle(
            'ScoreStyle',
            parent=normal_style,
            fontSize=14,
            textColor=score_color,
            fontName=FONT_NAME
        )
        elements.append(Paragraph(f"得分: {score}", score_style))
        elements.append(Spacer(1, 12))
        
        # Sections
        sections = [
            ("1. 原题目", "original_question"),
            ("2. 学生答案", "student_answer"),
            ("3. 标准答案", "standard_answer"),
            ("4. 知识点识别", "knowledge_points"),
            ("5. 解题思路", "solution_approach"),
            ("6. 错误分析", "error_analysis")
        ]
        
        for title, key in sections:
            elements.append(Paragraph(title, heading_style))
            content = str(data.get(key, 'N/A'))
            # Handle newlines in content for PDF
            content = content.replace('\n', '<br/>')
            elements.append(Paragraph(content, normal_style))
            elements.append(Spacer(1, 8))
            
        doc.build(elements)
