# 最终交付检查清单

## 文档与身份

- [x] 方案文档中的参赛队伍已填写为报名队伍名`X-M`。
- [x] 方案构建器已使用`--team-name 'X-M'`生成文档，身份占位符审计为0。
- [x] 已核对作品名称、完成日期、模型版本和技术栈。
- [ ] 在 Word/LibreOffice 中逐页检查 DOCX：无截断、溢出、异常分页或字体替换。
- [x] 方案和演示脚本只陈述 OOF/现场验收证据，不虚构测试集或榜单成绩。

## 运行与密钥

- [x] `.env`、`api key.txt`、访问令牌、终端历史和 SQLite 日志不在提交包中。
- [x] 四个`models/fold*.pt`已纳入代码目录，真实FastAPI `/health`已通过验收。
- [x] Qwen正常链和故障降级链均已完成端到端验收。
- [x] `start_ui.ps1`默认启动四折真实YOLO集成且页面明确显示运行模式；无Qwen密钥时仍可独立运行，Qwen通过显式开关启用。
- [x] Ruff、41项pytest、`pip check`全部通过。

## 真实结果

- [x] 仅由参赛者明确传入测试图片目录运行`generate_submission.ps1`。
- [x] `result/result.json`由四个YOLO分类模型对用户指定目录真实推理生成，不是样例或手工伪造文件。
- [x] `validate-submission`返回`valid=true`，共695条记录、0错误。
- [x] 文件名无重复，宽高、标签枚举和JSON编码符合官方Schema。

## 归档

- [x] 使用`build_delivery.ps1 -ConfirmRealResult`生成，不手工整目录压缩。
- [x] 构建阶段`validate-delivery`返回`valid=true`。
- [x] `.tar.gz`顶层仅有`code/`、`design/`、`result/`。
- [x] `result/result.json`存在；`design/`含最终方案；`code/`含四折权重与`MANIFEST.sha256`。
- [x] 已独立复核归档SHA-256，并保存到本次交付记录。
- [ ] 提交前再检查官方最新通知、每日最多5次限制和截止时间。
