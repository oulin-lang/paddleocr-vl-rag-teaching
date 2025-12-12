支持的文件格式与处理方式

当前系统仅支持以下类型的文件处理：

- 文本：`.txt`（通过 `file_management/txt_processor.py` 切分与索引）
- PTF：`.ptf`（通过 `file_management/ptf_processor.py` 解析与索引）
- 图片：`.jpg`、`.jpeg`、`.png`、`.bmp`、`.gif`（通过 `file_management/ptf_processor.py` OCR 解析与索引）

不支持的格式（请求将被拒绝）：

- Office 文档：`.doc`、`.docx`、`.xls`、`.xlsx`、`.ppt`、`.pptx`
- PDF：`.pdf`

后端接口行为（`Backend/app.py`）：

- 上传校验：仅允许 `.txt`、`.ptf`、常见图片扩展；超出大小或非法扩展将返回错误
- 病毒扫描：在 Windows 环境尝试调用 Defender 进行扫描，失败则跳过
- 索引与检索：
  - `txt` 走 `TxtHybridSearch`（BM25+向量，必要时降级）
  - `ptf`/图片走 `ptf_search`（OCR+向量检索，必要时降级）

路由策略（`Backend/rag_manage.py`）：

- `txt` → `TxtHybridSearch`
- `ptf` 与 `image` → `ptf_search`
- 其他 → 拒绝并返回“不支持的文件类型”
