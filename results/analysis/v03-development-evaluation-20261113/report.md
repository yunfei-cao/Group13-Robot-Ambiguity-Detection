# v0.3-candidate 开发集评估

**本次是开发集评估，不是新的独立盲测。** 76 对此前未运行样本仍来自开发集，不能称为独立测试集。

- Run: `group13-dev-v01-v03-20261113T150521Z-b2a4a074-candidate`
- 模型: `qwen-plus-2025-12-01`；Prompt: `v0.3-candidate`。
- 只读取 candidate 的510条真实预测；不依赖、不等待 v0.1 实验。
- 每种 Prompt 均严格按 sample_id 匹配170条指令和 Ground Truth，覆盖85对，每对包含两个不同标签。
- 三种策略均为170条唯一预测，重复0、漏配0、额外ID 0，有效输出170/170；总计510/510有效。
- 标签与 data/processed/dev_labels.csv 对应行完全一致；推理输入快照与开发集原始对应行及v0.3输入文件完全一致。
- 9/76分组来自 development_overlap.json，并与 development_pairs.csv、历史真实预测ID交集交叉核验。

## 计算口径

复用 WANG 的 evaluation_adapter.convert_predictions、evaluate.score_samples 和 calculate_metrics。
仅去除响应首尾空白，严格接受 CLEAR 或 NEEDS_CLARIFICATION；不修改预测或真实标签。
Precision、Recall 默认以 NEEDS_CLARIFICATION 为正类；CSV另含 CLEAR 类及宏平均 Precision/Recall。
Macro-F1 为两个类别F1的算术平均；Pair Accuracy 要求一对中两条预测均正确。
混淆矩阵行是真实标签、列是预测标签，顺序均为 [CLEAR, NEEDS_CLARIFICATION]；矩阵为 [[TN, FP], [FN, TP]]。
Accuracy、Macro-F1、Pair Accuracy 已用独立计数公式复核。CSV指标为0–1原始数值。

## 全部85对

| Prompt | 正确/样本 | Accuracy | Macro-F1 | Precision | Recall | 正确对/总对 | Pair Accuracy | 混淆矩阵 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Zero-shot | 98/170 | 57.6471% | 57.6236% | 58.0247% | 55.2941% | 16/85 | 18.8235% | [[51, 34], [38, 47]] |
| Few-shot | 96/170 | 56.4706% | 56.4646% | 56.3218% | 57.6471% | 15/85 | 17.6471% | [[47, 38], [36, 49]] |
| Checklist | 97/170 | 57.0588% | 56.9381% | 57.8947% | 51.7647% | 17/85 | 20.0000% | [[53, 32], [41, 44]] |

## 历史使用9对

| Prompt | 正确/样本 | Accuracy | Macro-F1 | Precision | Recall | 正确对/总对 | Pair Accuracy | 混淆矩阵 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Zero-shot | 12/18 | 66.6667% | 66.2500% | 63.6364% | 77.7778% | 3/9 | 33.3333% | [[5, 4], [2, 7]] |
| Few-shot | 11/18 | 61.1111% | 60.9907% | 60.0000% | 66.6667% | 2/9 | 22.2222% | [[5, 4], [3, 6]] |
| Checklist | 13/18 | 72.2222% | 72.1362% | 70.0000% | 77.7778% | 4/9 | 44.4444% | [[6, 3], [2, 7]] |

## 此前未运行76对

| Prompt | 正确/样本 | Accuracy | Macro-F1 | Precision | Recall | 正确对/总对 | Pair Accuracy | 混淆矩阵 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Zero-shot | 86/152 | 56.5789% | 56.5112% | 57.1429% | 52.6316% | 13/76 | 17.1053% | [[46, 30], [36, 40]] |
| Few-shot | 85/152 | 55.9211% | 55.9191% | 55.8442% | 56.5789% | 13/76 | 17.1053% | [[42, 34], [33, 43]] |
| Checklist | 84/152 | 55.2632% | 55.0687% | 56.0606% | 48.6842% | 13/76 | 17.1053% | [[47, 29], [39, 37]] |

## 结论与产物

全85对上，Zero-shot 表现最好：Accuracy 57.6471%，Macro-F1 57.6236%。该结论限于本次开发集结果。

- metrics.csv：3个分组 × 3种策略的完整指标和混淆矩阵计数。
- scored_predictions.csv：510条逐样本匹配、原始响应、预测、GT、环境与指令。
- pair_results.csv：85对 × 3种策略的逐对完整结果，包含历史使用标记。
- coverage_validation.csv：每种Prompt的重复、漏配及有效性检查。
- audit.json：输入来源、SHA-256、分组清单与评估函数。
- evaluate_offline.py：离线复现脚本；使用排他创建，拒绝覆盖已有输出。

本次未调用Qwen API，未修改预测、Ground Truth或CAI Prompt，未覆盖历史实验结果，未Commit、未Push。
受保护源文件在评估前后的SHA-256完全一致。
