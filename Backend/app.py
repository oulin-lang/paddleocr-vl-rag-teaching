""" 
FastAPI 应用入口 
 
提供三类接口： 
- 文件处理：上传 
- 作业批改工作流：批量执行与结果查询 
- 会话管理：问候、加文件、聊天、历史、清理 
 
说明： 
- 代码在缺少第三方依赖时做了降级处理；运行前请安装 fastapi/uvicorn 等依赖 
""" 
 
import os 
import time 
import uuid 
from typing import Dict, Any, List
 
try: 
    from fastapi import FastAPI, UploadFile, File, Body, BackgroundTasks
    from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware 
except Exception: 
    # 缺少 fastapi 依赖时占位，避免导入报错 
    FastAPI = None 
    JSONResponse = None 
    StreamingResponse = None
    FileResponse = None
    StaticFiles = None
    CORSMiddleware = None
    UploadFile = None
    File = None
    Body = None 
 
# Rate limit data: session_id -> list of timestamps 
_rate_limit_data = {} 
_upload_meta: Dict[str, Any] = {}
_file_index: Dict[str, Any] = {}
_session_files: Dict[str, list] = {}
_grading_tasks: Dict[str, Any] = {}
MAX_UPLOAD_SIZE = 100 * 1024 * 1024
PROCESS_TIMEOUT = 600
 
def _check_mcp_rate_limit(session_id: str, max_requests: int = 5, window_seconds: int = 60) -> bool: 
    """检查MCP服务调用速率限制""" 
    now = time.time() 
    if session_id not in _rate_limit_data: 
        _rate_limit_data[session_id] = [] 
 
    # 清理过期时间戳 
    _rate_limit_data[session_id] = [t for t in _rate_limit_data[session_id] if now - t < window_seconds] 
 
    if len(_rate_limit_data[session_id]) >= max_requests: 
        return False 
 
    _rate_limit_data[session_id].append(now) 
    return True 
 
def _clear_rate_limit(session_id: str): 
    """清理会话的限流数据""" 
    if session_id in _rate_limit_data: 
        del _rate_limit_data[session_id] 
 
try: 
    from session_manager import SessionManager 
    from runtime_config import runtime_config 
except Exception as e: 
    print(f"Error importing SessionManager: {e}")
    # 缺少内部依赖时占位，接口将返回可读错误 
    SessionManager = None 
    runtime_config = None 
 
try: 
    from workflow.correction_workflow import CorrectionWorkflow 
except Exception: 
    # 工作流缺失时占位 
    CorrectionWorkflow = None 

try:
    from workflow.grading_workflow import GradingWorkflow
except Exception:
    GradingWorkflow = None
 
def _ensure_dir(path: str): 
    """确保目录存在""" 
    os.makedirs(path, exist_ok=True) 
 
def _save_upload(session_id: str, up: UploadFile) -> str: 
    """保存上传文件到会话目录""" 
    # 将上传文件保存到 Backend/storage/uploads/session_id 目录
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "uploads", session_id) 
    _ensure_dir(root) 
    fn = os.path.join(root, up.filename) 
    with open(fn, "wb") as f: 
        f.write(up.file.read()) 
    return fn 

def _get_file_size(path: str) -> int:
    """获取文件大小，出错返回0"""
    try:
        return os.path.getsize(path)
    except Exception:
        return 0

def _get_ext(path: str) -> str:
    """获取文件扩展名（小写）"""
    return os.path.splitext(path)[1].lower()

def _get_type(ext: str) -> str:
    """根据扩展名判断文件类型"""
    if ext in [".pdf"]:
        return "pdf"
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
        return "image"
    if ext in [".txt"]:
        return "txt"
    if ext in [".ptf"]:
        return "ptf"
    if ext in [".doc", ".docx"]:
        return "word"
    if ext in [".xls", ".xlsx"]:
        return "excel"
    if ext in [".ppt", ".pptx"]:
        return "ppt"
    return "unknown"

def _validate_file(path: str) -> Dict[str, Any]:
    """验证文件是否符合要求"""
    size = _get_file_size(path)
    ext = _get_ext(path)
    allowed = {".pdf", ".ptf", ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".txt"}
    ok = ext in allowed and size > 0 and size <= MAX_UPLOAD_SIZE
    return {"ok": ok, "size": size, "ext": ext, "type": _get_type(ext)}

def _scan_file(path: str, timeout_sec: int = 10) -> bool:
    """调用Windows Defender扫描文件"""
    try:
        import shutil, subprocess
        defender = shutil.which("MpCmdRun.exe")
        if defender:
            p = subprocess.run([defender, "-Scan", "-ScanType", "3", "-File", path], timeout=timeout_sec)
            return p.returncode == 0
        return True
    except Exception:
        return True

def _summarize_result(res: Any) -> Dict[str, Any]:
    """从处理结果中提取摘要信息"""
    try:
        if isinstance(res, list) and res:
            it = res[0]
            return {
                "status": it.get("status"),
                "message": it.get("message"),
                "count": it.get("count") or it.get("elements_count") or 0
            }
        if isinstance(res, dict):
            return {
                "status": res.get("status"),
                "message": res.get("message"),
                "count": res.get("count") or res.get("elements_count") or 0
            }
    except Exception:
        pass
    return {}

def _register_file(session_id: str, path: str, meta: Dict[str, Any], result: Any = None) -> str:
    """注册文件到内存索引"""
    try:
        import uuid as _uuid
        fid = f"file_{_uuid.uuid4().hex}{_get_ext(path)}"
        item = {
            "file_id": fid,
            "session_id": session_id,
            "name": os.path.basename(path),
            "path": path,
            "size": meta.get("size", 0),
            "type": meta.get("type", "unknown"),
            "created": time.time(),
            "status": "indexed",
            "summary": _summarize_result(result),
        }
        _file_index[fid] = item
        _session_files.setdefault(session_id, []).append(fid)
        return fid
    except Exception:
        return ""

async def _run_grading_task(task_id: str, file_list: List[str], output_dir: str):
    """
    运行批改任务的后台异步函数
    
    Args:
        task_id: 任务ID
        file_list: 待批改文件列表
        output_dir: 结果输出目录
    """
    try:
        if GradingWorkflow is None:
            raise ImportError("GradingWorkflow module not available")
            
        workflow = GradingWorkflow(output_dir=output_dir)
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        generated_files = await workflow.process_files(file_list)
        
        # Update task status
        if task_id in _grading_tasks:
            _grading_tasks[task_id]["status"] = "completed"
            _grading_tasks[task_id]["results"] = generated_files
        
    except Exception as e:
        print(f"Grading task {task_id} failed: {e}")
        import traceback
        traceback.print_exc()
        if task_id in _grading_tasks:
            _grading_tasks[task_id]["status"] = "failed"
            _grading_tasks[task_id]["error"] = str(e)

if FastAPI is not None:  
    # 创建 FastAPI 应用实例 
    app = FastAPI() 
    if CORSMiddleware: 
        app.add_middleware( 
            CORSMiddleware, 
            allow_origins=["*"], 
            allow_credentials=True, 
            allow_methods=["*"], 
            allow_headers=["*"], 
        ) 
    
    if StaticFiles:
        # 静态文件挂载路径修正为 Backend/storage/uploads
        uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

        # 批改结果挂载路径修正为 Backend/storage/output
        grading_outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "output")
        os.makedirs(grading_outputs_dir, exist_ok=True)
        app.mount("/grading_outputs", StaticFiles(directory=grading_outputs_dir), name="grading_outputs")
else: 
    app = None 
 
sm = SessionManager() if SessionManager else None  # 会话管理器实例 
 
if app is not None: 
    @app.get("/health") 
    def health() -> Dict[str, Any]: 
        """健康检查""" 
        return {"status": "ok"} 
 
    @app.post("/sessions") 
    def create_session(payload: Dict[str, Any] = Body(None)): 
        """创建新会话并初始化""" 
        if sm is None: 
            return JSONResponse({"error": "session_manager_unavailable"}, status_code=500) 
        sid = (payload or {}).get("session_id") or f"session_{uuid.uuid4()}" 
        greeting = sm.greet(sid) 
        return {"session_id": sid, "message": greeting} 
 
    @app.get("/sessions") 
    def list_sessions(): 
        """列出所有会话信息""" 
        if sm is None: 
            return JSONResponse({"error": "session_manager_unavailable"}, status_code=500) 
        return sm.list_all_sessions() 
 
    @app.delete("/sessions/{session_id}") 
    def delete_session(session_id: str): 
        """删除会话（清理缓存与记忆）""" 
        if sm is None: 
            return JSONResponse({"error": "session_manager_unavailable"}, status_code=500) 
        sm.cleanup_session(session_id) 
        _clear_rate_limit(session_id) 
        return {"status": "ok", "session_id": session_id} 
 
    @app.get("/sessions/{session_id}/history") 
    def history(session_id: str): 
        """获取会话历史记录""" 
        if sm is None: 
            return JSONResponse({"error": "session_manager_unavailable"}, status_code=500) 
        return sm.get_session_history(session_id) 
 
    @app.post("/sessions/{session_id}/chat") 
    async def chat(session_id: str, payload: Dict[str, Any] = Body(...)):
        """基于RAG的智能聊天（文档不足时联网）- 支持流式输出"""
        if sm is None:
            return JSONResponse({"error": "session_manager_unavailable"}, status_code=500)

        q = payload.get("question") or ""
        enable_net_search = payload.get("net_search", False)
        # 默认启用流式输出，除非客户端显式关闭
        stream = payload.get("stream", True)
        file_paths = payload.get("file_paths")
 
        # 联网功能需进行限流和权限验证 
        if enable_net_search: 
            # 1. 检查是否启用了MCP访问权限 
            if runtime_config and not runtime_config.enable_mcp_access: 
                return JSONResponse({"error": "MCP/Network functions disabled"}, status_code=403) 
 
            # 2. 速率限制检查 
            if not _check_mcp_rate_limit(session_id): 
                return JSONResponse({"error": "Rate limit exceeded for MCP services (5 req/min)"}, status_code=429) 
 
        if stream:
            async def sse_generator():
                print(f"[Debug] SSE Generator started for {session_id}")
                import json
                try:
                    async for chunk in sm.chat_with_rag_stream(session_id, q, enable_net_search=enable_net_search, file_paths=file_paths):
                        # 构造SSE格式数据
                        data = json.dumps({"content": chunk}, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                    print(f"[Debug] SSE Generator sending [END]")
                    yield "data: [END]\n\n"
                except Exception as e:
                    print(f"[Debug] SSE Generator Error: {e}")
                    error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
                    yield f"data: {error_data}\n\n"
                finally:
                    print(f"[Debug] SSE Generator finished")

            return StreamingResponse(sse_generator(), media_type="text/event-stream")
        else:
            return await sm.chat_with_rag(session_id, q, enable_net_search=enable_net_search, file_paths=file_paths)
 
    @app.post("/sessions/{session_id}/add-file") 
    async def add_file(session_id: str, file: UploadFile = File(None), file_path: str | None = Body(None)): 
        """向会话添加文件（上传或路径）""" 
        if sm is None: 
            return JSONResponse({"error": "session_manager_unavailable"}, status_code=500) 
        if file is not None: 
            p = _save_upload(session_id, file) 
            return await sm.add_document_to_session(session_id, p) 
        if file_path: 
            return await sm.add_document_to_session(session_id, file_path) 
        return JSONResponse({"error": "no_file"}, status_code=400) 
 
    @app.post("/files/upload") 
    async def upload_file(session_id: str, file: UploadFile = File(...), skip_summary: bool = False): 
        if sm is None: 
            return JSONResponse({"error": "session_manager_unavailable"}, status_code=500) 
        path = _save_upload(session_id, file) 
        meta = _validate_file(path) 
        if not meta["ok"]: 
            try:
                os.remove(path)
            except Exception:
                pass
            return JSONResponse({"error": "invalid_file", "meta": meta}, status_code=400) 
        scanned = _scan_file(path) 
        if not scanned: 
            try:
                os.remove(path)
            except Exception:
                pass
            return JSONResponse({"error": "virus_detected", "meta": meta}, status_code=400) 
        res = await sm.add_document_to_session(session_id, path, skip_summary=skip_summary) 
        summary = _summarize_result(res) 
        fid = _register_file(session_id, path, meta, res)
        return {"status": "completed", "session_id": session_id, "file_id": fid, "file": os.path.basename(path), "size": meta["size"], "type": meta["type"], "result": res, "summary": summary} 

    @app.post("/files/upload/async")
    async def upload_file_async(session_id: str, file: UploadFile = File(...), background: BackgroundTasks = None, skip_summary: bool = False):
        if sm is None:
            return JSONResponse({"error": "session_manager_unavailable"}, status_code=500)
        path = _save_upload(session_id, file)
        meta = _validate_file(path)
        if not meta["ok"]:
            try:
                os.remove(path)
            except Exception:
                pass
            return JSONResponse({"error": "invalid_file", "meta": meta}, status_code=400)
        scanned = _scan_file(path)
        if not scanned:
            try:
                os.remove(path)
            except Exception:
                pass
            return JSONResponse({"error": "virus_detected", "meta": meta}, status_code=400)
        task_id = sm.submit_file_task(session_id, path, skip_summary=skip_summary)
        fid = f"file_{uuid.uuid4().hex}{_get_ext(path)}"
        _upload_meta[task_id] = {"status": "processing", "session_id": session_id, "file_id": fid, "file": os.path.basename(path), "size": meta["size"], "type": meta["type"], "created": time.time()}
        def _track():
            import time as _t
            end = _t.time() + PROCESS_TIMEOUT
            while _t.time() < end:
                st = sm.get_file_task_status(task_id)
                s = st.get("status")
                if s in {"completed", "failed"}:
                    res = st.get("result")
                    print(f"[Debug] File task {task_id} completed. Result type: {type(res)}")
                    
                    summary_val = None
                    if s == "completed":
                        try:
                            fid2 = _register_file(session_id, path, meta, res)
                            _upload_meta[task_id]["file_id"] = fid2 or _upload_meta[task_id]["file_id"]
                            
                            # Lift summary_text from result if available
                            summary_val = None
                            if res and isinstance(res, list) and len(res) > 0:
                                item = res[0]
                                print(f"[Debug] First item keys: {item.keys() if isinstance(item, dict) else 'not dict'}")
                                if isinstance(item, dict) and "summary_text" in item:
                                    summary_val = item["summary_text"]
                                    print(f"[Debug] Found summary_text length: {len(summary_val) if summary_val else 0}")
                                    _upload_meta[task_id]["summary_text"] = summary_val
                            
                            # Update meta again with summary to ensure consistency
                            if summary_val:
                                _upload_meta[task_id]["summary"] = _summarize_result(res) # Keep original summary structure
                                _upload_meta[task_id]["summary_text"] = summary_val

                        except Exception as e:
                            print(f"[Debug] Error in _track processing: {e}")
                            pass
                    
                    # Update status LAST to ensure summary is available when status becomes 'completed'
                    # But we already updated status at the beginning of the block.
                    # We should update again to ensure 'summary_text' is visible in the final atomic read if possible,
                    # though in Python dict operations are atomic-ish, the previous update already set 'completed'.
                    # Ideally, we should construct the full update dict first.
                    
                    # Correct approach: Update status and summary together if possible, 
                    # but since we need 'res' to get summary, and 'res' comes from 'st', 
                    # we can just update _upload_meta with everything at once.
                    
                    # Re-update with full data
                    update_data = {
                        "status": s, 
                        "result": res, 
                        "error": st.get("error"), 
                        "finished": _t.time(), 
                        "summary": _summarize_result(res)
                    }
                    if summary_val:
                        update_data["summary_text"] = summary_val
                    
                    _upload_meta[task_id].update(update_data)
                    break
                
                _t.sleep(0.5)
        if background is not None:
            background.add_task(_track)
        return {"task_id": task_id, "status": "processing", "meta": _upload_meta[task_id]}

    @app.get("/files/status/{task_id}")
    def get_file_status(task_id: str):
        if sm is None:
            return JSONResponse({"error": "session_manager_unavailable"}, status_code=500)
        st = sm.get_file_task_status(task_id)
        meta = _upload_meta.get(task_id)
        if meta:
            out = {"task": st, "meta": meta}
        else:
            out = {"task": st}
        return out

    @app.get("/files/subscribe/{task_id}")
    def subscribe_file_status(task_id: str):
        if sm is None:
            return JSONResponse({"error": "session_manager_unavailable"}, status_code=500)
        import json
        async def gen():
            import asyncio
            done = False
            while not done:
                st = sm.get_file_task_status(task_id)
                meta = _upload_meta.get(task_id)
                payload = {"task": st, "meta": meta}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                s = st.get("status")
                if s in {"completed", "failed", "not_found"}:
                    done = True
                    yield "data: [END]\n\n"
                    break
                await asyncio.sleep(0.5)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/files/meta/{file_id}")
    def get_file_meta(file_id: str):
        item = _file_index.get(file_id)
        if not item:
            return JSONResponse({"error": "not_found", "file_id": file_id}, status_code=404)
        return item

    @app.get("/files/list/{session_id}")
    def list_session_files(session_id: str):
        fids = _session_files.get(session_id, [])
        items = [ _file_index.get(fid) for fid in fids if fid in _file_index ]
        return {"session_id": session_id, "files": items}
 
    @app.post("/workflow/run") 
    def run_workflow(session_id: str, payload: Dict[str, Any] = Body(...)): 
        """执行作业批改工作流""" 
        files = payload.get("file_list") or [] 
        if sm is None: 
            return JSONResponse({"error": "session_manager_unavailable"}, status_code=500) 
        try: 
            res = sm.run_homework_workflow(session_id, files) 
            return res 
        except Exception as e: 
            return JSONResponse({"error": str(e)}, status_code=500) 
 
    @app.get("/workflow/progress/{session_id}") 
    def get_workflow_progress(session_id: str): 
        """查询工作流进度""" 
        if sm is None: 
            return JSONResponse({"error": "session_manager_unavailable"}, status_code=500) 
        return sm.get_homework_progress(session_id) 
 
    @app.get("/workflow/result/{session_id}") 
    def get_workflow_result(session_id: str): 
        """查询工作流输出文件列表""" 
        base = os.path.join(os.getcwd(), "agent_rag", "Homework_Summary", session_id) 
        if not os.path.exists(base): 
            return {"outputs": []} 
        items = [] 
        for name in os.listdir(base): 
            items.append(os.path.join(base, name)) 
        return {"outputs": items} 
 
    @app.post("/grading/batch")
    async def start_batch_grading(
        background_tasks: BackgroundTasks,
        payload: Dict[str, Any] = Body(...)
    ):
        """
        启动批量作业批改任务
        Payload:
        {
            "session_id": "...",
            "file_list": ["path/to/file1", "path/to/file2"] 
        }
        """
        if GradingWorkflow is None:
            return JSONResponse({"error": "GradingWorkflow not available"}, status_code=500)
            
        session_id = payload.get("session_id")
        file_list = payload.get("file_list", [])
        file_ids = payload.get("file_ids", [])
        
        # Resolve file_ids to paths if provided
        if file_ids:
            for fid in file_ids:
                if fid in _file_index:
                    file_list.append(_file_index[fid]["path"])
        
        if not session_id or not file_list:
             return JSONResponse({"error": "Missing session_id or file_list/file_ids"}, status_code=400)

        task_id = str(uuid.uuid4())
        # Use storage/output/task_id to match StaticFiles mount
        base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "output")
        output_dir = os.path.join(base_output_dir, task_id)
        
        _grading_tasks[task_id] = {
            "status": "processing",
            "session_id": session_id,
            "output_dir": output_dir,
            "results": [],
            "created_at": time.time()
        }
        
        background_tasks.add_task(_run_grading_task, task_id, file_list, output_dir)
        
        return {"task_id": task_id, "status": "processing"}

    @app.get("/grading/status/{task_id}")
    def get_grading_status(task_id: str):
        """查询批改任务状态及结果"""
        task = _grading_tasks.get(task_id)
        if not task:
            return JSONResponse({"error": "Task not found"}, status_code=404)
        
        # Construct public URLs for results
        results = []
        if task["status"] == "completed":
            base_url = "/grading_outputs/" + task_id + "/"
            for file_path in task["results"]:
                filename = os.path.basename(file_path)
                results.append({
                    "filename": filename,
                    "url": base_url + filename,
                    "path": file_path
                })
                
        return {
            "task_id": task_id,
            "status": task["status"],
            "created_at": task["created_at"],
            "results": results,
            "error": task.get("error")
        }

    @app.get("/grading/download/{task_id}")
    def download_grading_results(task_id: str):
        """下载批改结果打包(Zip)"""
        task = _grading_tasks.get(task_id)
        if not task or task["status"] != "completed":
            return JSONResponse({"error": "Task not ready or not found"}, status_code=404)
            
        output_dir = task["output_dir"]
        zip_filename = f"grading_results_{task_id}.zip"
        zip_path = os.path.join(output_dir, zip_filename)
        
        if not os.path.exists(zip_path):
            import zipfile
            try:
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            if file == zip_filename:
                                continue
                            if not file.lower().endswith('.pdf'):
                                continue
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, output_dir)
                            zipf.write(file_path, arcname)
            except Exception as e:
                return JSONResponse({"error": f"Failed to zip files: {str(e)}"}, status_code=500)
                    
        return FileResponse(zip_path, filename=zip_filename)

def _run(): 
    """本地启动入口""" 
    import uvicorn 
    uvicorn.run(app=app, host="127.0.0.1", port=8003, reload=False) 

if __name__ == "__main__": 
    _run() 
