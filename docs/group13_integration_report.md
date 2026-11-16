# Group 13 三人代码联调报告

分支 `feature/model-experiment`，基线 HEAD `3bf202a`。仅离线 Mock；未调用真实 Qwen、未执行开发集或正式批量实验、未 Commit/Push/切换分支。CAI 和 WANG 原始代码与数据均未修改。

## 实际接口

- 输入 CSV：`sample_id,pair_id,environment,instruction`。适配为内部 `id,pair_id,environment,instruction`。
- 标签 CSV：`sample_id,pair_id,gold_label,category,source_record,source_index,source_id,instruction_column`，仅独立评估转换阶段读取。
- CAI：`build_messages(strategy, environment, instruction, examples=None)` 返回 system/user 消息列表；`build_prompt(...)` 返回单文本。本次使用前者。
- 策略：`zero_shot,few_shot,checklist`。版本 `v0.1-dev`；清单仍为 DRAFT。
- 固定 Few-shot：开发配对 dev_0/dev_8/dev_63/dev_36，共 4 条、每类 2 条；不动态选择测试样本。
- WANG：`score_samples(samples)` 要求 sample_id/pair_id/gold_label/raw_response/request_status；完整配对、样本 ID 唯一。`calculate_metrics(scored_samples)` 计算指标。

## 修改

- `src/run_experiment.py`：严格 CSV 输入适配；默认 Mock 也使用真实 Prompt；builder 只收到 environment/instruction；增加 --limit；最终预测补充 raw_text、Token、模型和延迟信息。
- `src/prompt_adapter.py`：真实构建函数桥接；固定示例哈希、顺序、版本校验。Windows CRLF 规范化到 LF 后校验，未改原文件。
- `src/evaluation_adapter.py`：推理完成后独立连接标签、转换状态并按策略分组；拒绝重复、缺失策略、不完整配对及旧版缺少 raw_text 的预测。
- `tests/test_run_experiment.py`：既有路由测试显式指定 fixture builder。
- `tests/test_integration.py`：新增 7 项真实接口集成测试。
- `docs/run_experiment.md`：更新使用约定。

## 验证结果

`python -m unittest discover -s tests -q`：62 项通过，包括原有 55 项及新增 7 项。覆盖真实三种消息、输入白名单、CSV BOM/引号/多行/异常字段、日志、重复运行、错误重试与中断恢复、无效输出、Token 未知值、凭据脱敏、示例完整性及 WANG 评分接口。测试禁止 socket.connect。

实际命令（运行两次）：

```powershell
python -m src.run_experiment --input data/processed/test_inputs.csv --limit 2 --run-id group13-integration-3bf202a
```

使用前两条指令 s_24222829effb16884276 / s_e51e9f4a55668d6bbea4，均属于 test_104。首次 6 个逻辑请求、6 次 Mock 调用、6 条有效最终预测、12 条尝试事件；第二次 0 次新增调用。Mock 固定返回 CLEAR；Token 三项均 null，不伪造用量。

产物在 `results/mock/group13-integration-3bf202a/`：run_config.json、attempts.jsonl、predictions.jsonl，以及三份 evaluation_<strategy>.json。评估转换已实际运行，每策略 2 样本/1 完整配对；仅供接口验收。该目录被 Git 忽略，保留在本机。

## 剩余事项

- CAI 的 Prompt 仍为草稿、数据清单的人工语义审查仍待完成；本次未作性能结论或正式冻结。
- WANG 入口 main 仍使用其模拟样本；请通过独立 convert_predictions 返回的策略列表调用 score_samples/calculate_metrics。
- 旧 predictions.jsonl 无 raw_text，适配器明确拒绝，需要另行从 attempts 做显式迁移；未改旧日志。
- 本次未验证真实网络服务、真实 Token 用量或模型准确率。
