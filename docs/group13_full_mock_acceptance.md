# 全量 Mock 技术验收

结论：当前 VS Code 工作区通过正式实验前的离线技术验收。此结论覆盖数据接口、Prompt 路由、日志、恢复去重与评估交接，不代表真实模型效果或真实服务可用性。

- 分支 feature/model-experiment；HEAD 3bf202a。保留此前所有未提交修改，未改动已有 src、tests、数据或 Prompt 文件；执行验收时对这些文件逐一比对 SHA-256。
- 输入 data/processed/test_inputs.csv：200 条、100 个完整配对。
- 调用 CAI 真实 build_messages；Zero-shot/Few-shot/Checklist 各 200 次，共 600 个 Mock 逻辑请求。
- 600 条唯一 logical_id，完整覆盖全部 (sample_id,strategy)；600 条有效预测；未解决请求 0。
- attempts.jsonl 共 1200 条事件，600 组 started/completed，均为第一次尝试；身份、Prompt 哈希、状态、模型、延迟、Token 字段完整。
- 重复运行新增调用 0；run_config、attempts、predictions 的字节哈希均未变化。
- 三策略经 convert_predictions 连接标签后，各 200 条/100 个完整配对，通过 WANG 的 score_samples 格式校验。
- 逐条比对送入 Mock 客户端的消息与 CAI 官方函数输出；user 消息仅含 environment/instruction。标签在推理和重复运行完成后才由评估适配器读取。
- 网络 socket.connect 和 ModelClient._request 在全量验收中被阻断；真实调用 0。检查运行产物未包含凭据类环境变量值，日志未保存完整输入环境或指令。
- 全部单元测试：python -m unittest discover -s tests -v，66 项通过（原 62 项 + 新增评估适配器负向校验 4 项）。覆盖输入白名单、敏感信息脱敏、无效输出、错误重试、中断、日志修复、状态转换等。
- Prompt 清单保持 DRAFT: development pilot pending；固定 Few-shot 不变。

本次新增 scripts/mock_acceptance.py、tests/test_evaluation_adapter.py 和本报告。未调用真实 API，未 Commit/Push。

运行证据：results/mock/group13-full-mock-acceptance/acceptance_summary.json；同目录保存 run_config.json、attempts.jsonl、predictions.jsonl 及 evaluation_zero_shot.json / evaluation_few_shot.json / evaluation_checklist.json。该目录被 Git 忽略。

复现新验收：python -m scripts.mock_acceptance --run-id <新的运行ID>。脚本会运行全量 Mock、立即重复并核验零新增；首次验收必须使用未运行过的 ID。已有产物保持原样。

剩余风险：

1. Mock 固定返回 CLEAR，不检验语义准确率；Token 为 null，不能推断真实成本、延迟或截断率。
2. 真实 API 的鉴权、模型可用性、配额、限流及服务响应未在线验证；离线测试只验证请求与响应处理约定。
3. Prompt 仍为 DRAFT，正式实验前需由组员完成版本审查和冻结；本次不改变状态。200 条测试指令按用户确认使用，不重新提出数据语义审查阻塞。
4. 中断后未知请求不会自动重发，真实实验出现此情况仍需独立核对与显式补跑，不能保证跨进程/网络恰好一次执行。
5. 脱敏覆盖已知密钥及常见凭据格式，不保证识别任意编码或混淆的敏感输出。正式产物分享前仍需检查。
6. 评估适配器允许完整配对子集；全量覆盖由本次验收脚本额外断言。旧版无 raw_text 的预测需要显式迁移。


??????66 ?????????????????? ID
`group13-precommit-full-mock-20261110` ?? 600 ? Mock ????????? 0 ??
600 ????????1200 ?????????? 200 ?/100 ??????? 0?
??????? `results/mock/group13-precommit-full-mock-20261110/`????????
???? Commit/Push???????????????????????????
