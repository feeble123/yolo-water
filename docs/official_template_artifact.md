# 官方方案模板蒸馏契约

## Reference

- 文件：`E:\GPT-Codex\Yolo-water\水域综合异常识别\AI智能体设计方案.docx`
- SHA-256：`8e8148db47a1fd7050c11027105f85299c72596ffaefd4c9a700824afc9877d0`
- 节数：1
- 页数：当前环境缺少 LibreOffice/soffice，无法通过标准渲染器核验
- 结构证据：`scripts/inspect_docx_template.py`、documents skill 的 `section_audit.py`、`style_lint.py`、`heading_audit.py`、`images_audit.py`、`fields_report.py` 和 `content_controls.py`

## Page system

- A4 纵向：8.267 × 11.694 英寸。
- 页边距：左/右 1.25 英寸，上/下 1.00 英寸。
- 页眉/页脚距离：各 0.50 英寸。
- 单节、无首页差异、无奇偶页差异、无分栏。

## Typography and components

- 标题：普通段落，Arial 26 pt，加粗、居中，段前/段后各 24 pt，1.2 倍行距。
- 元数据：参赛队伍、作品名称、完成日期各一行；字段名 Arial 加粗，段前/段后各 6 pt，1.2 倍行距。
- 一级章节：普通段落，中文序号，Arial 16 pt，加粗，段前 16 pt、段后 6 pt，1.2 倍行距。
- 正文基线：Arial，段前/段后各 6 pt，1.2 倍行距。
- 模板无表格、图片、页眉页脚内容、域、内容控件、编号定义或真实 Heading 样式。

## Content flow and slot map

1. 标题：保留“AI智能体设计方案”。
2. 参赛队伍：用户未提供队伍名，保留显式待填槽位，不自行虚构。
3. 作品名称：填写“水域智巡——基于YOLO与Qwen的水域综合异常识别智能体”。
4. 完成日期：填写当前交付日期。
5. 一、需求与痛点分析：替换其后空白段落并扩展为业务背景、目标用户、核心痛点和需求边界。
6. 二、智能体总体架构设计：替换其后空白段落并扩展为架构、流程、接口与合规设计。
7. 三、核心模块详细设计：替换其后空白段落并扩展为视觉工具、Agent编排、Qwen解释、服务层、审计与提交链。
8. 四、创新点设计：替换其后空白段落并扩展为可验证创新点，不夸大模型能力。
9. 五、效果评估：保留“评估结果”和“任务执行耗时”的语义，填入四折 OOF 与 Phase 4 现场验收证据，并新增风险边界。

## Package preservation

模板共 14 个包部件：内容类型、根关系、document.xml、document关系、脚注、尾注、主题、设置、样式、Web设置、字体表、core/app/custom 属性。除 `word/document.xml` 和必要的文档属性外均视为 preserve-only；不添加宏、外部关系、OLE、嵌入文件或远程图片。

## Fidelity gates

- 原始模板必须保持上述哈希不变。
- 最终文档保持单节 A4 页面几何、标题/元数据/五章顺序和模板字体节奏。
- 新增内容不得包含 API Key、测试图片、测试集路径、虚构榜单成绩或“目标检测框”等不实表述。
- 在可用 Word/LibreOffice 环境中打开后须人工逐页检查截断、表格溢出、分页和字体替换；当前环境缺少 `soffice`，该视觉门保留为交付待办。
