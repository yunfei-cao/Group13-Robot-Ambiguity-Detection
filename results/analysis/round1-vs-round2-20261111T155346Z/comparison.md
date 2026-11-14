# Group 13：两轮 Qwen Plus Pilot 离线评估与重复性分析

两轮使用**相同的 200 条测试指令（100 对、CLEAR/NEEDS_CLARIFICATION 各 100 条）**、相同的三种策略和 **v0.1-dev DRAFT Prompt**。这是 Pilot 重复性实验，不是最终冻结实验，也不是独立测试集复验。未依据测试结果修改 Prompt。

Round 1：`group13-full-pilot-qwen200-20261111T090248Z-76877106`；Round 2：`group13-full-pilot-round2-20261111T153626Z-bc85f07d`。

## 主要指标

下表为每策略分别评估；Precision/Recall 使用两个金标准类别的宏平均，CSV 同时提供每类 Precision/Recall/F1。所有数值为比例，表中转为百分比。

| 轮次 | 策略 | 口径 | n | Accuracy | Macro-F1 | Precision(macro) | Recall(macro) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| round1 | Zero-shot | all_jobs | 200 | 53.50% | 53.50% | 53.50% | 53.50% |
| round1 | Zero-shot | valid_only | 200 | 53.50% | 53.50% | 53.50% | 53.50% |
| round1 | Few-shot | all_jobs | 200 | 56.00% | 53.67% | 57.80% | 56.00% |
| round1 | Few-shot | valid_only | 199 | 56.28% | 53.81% | 57.80% | 56.17% |
| round1 | Checklist | all_jobs | 200 | 55.50% | 55.50% | 55.50% | 55.50% |
| round1 | Checklist | valid_only | 200 | 55.50% | 55.50% | 55.50% | 55.50% |
| round2 | Zero-shot | all_jobs | 200 | 53.50% | 53.50% | 53.50% | 53.50% |
| round2 | Zero-shot | valid_only | 200 | 53.50% | 53.50% | 53.50% | 53.50% |
| round2 | Few-shot | all_jobs | 200 | 55.00% | 52.92% | 56.07% | 55.00% |
| round2 | Few-shot | valid_only | 200 | 55.00% | 52.92% | 56.07% | 55.00% |
| round2 | Checklist | all_jobs | 200 | 54.00% | 54.00% | 54.00% | 54.00% |
| round2 | Checklist | valid_only | 200 | 54.00% | 54.00% | 54.00% | 54.00% |

使用 `src.evaluation_adapter.convert_predictions` 连接 `data/processed/test_labels.csv`，在每策略完整配对上调用 WANG 的 `score_samples`，再用 `calculate_metrics` 计算。valid_only 在完成评分后筛选，不会因一条失败丢弃其有效配对样本。独立计数公式交叉核验 Accuracy/Macro-F1。

all_jobs：失败/无效输出计错，且只给真实类增加 FN，不虚构另一类 FP；Macro-F1 始终只平均两个真实类别。第一轮 600 条最终记录中 599 条有效，Few-shot 的一条历史 network_error 保留。valid_only：第一轮三策略分别为 200/199/200 条（合计 599），第二轮均为 200（合计 600）。

CSV 还保留 WANG 的漏检率、不必要澄清率、完整配对正确率及有效率。第一轮 Few-shot valid_only 有一个不完整配对，其配对指标留空；ALL 为跨策略池化描述，配对指标也留空。

## 两轮预测一致率

| 策略 | 相同/双方有效 | 一致率 | 双方有效覆盖率 | 全量相同有效标签率 | 标签翻转 | 有效性变化 | 对→错 | 错→对 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | 200/200 | 100.00% | 100.00% | 100.00% | 0 | 0 | 0 | 0 |
| Few-shot | 191/199 | 95.98% | 99.50% | 95.50% | 8 | 1 | 5 | 3 |
| Checklist | 195/200 | 97.50% | 100.00% | 97.50% | 5 | 0 | 4 | 1 |
| ALL | 586/599 | 97.83% | 99.83% | 97.67% | 13 | 1 | 9 | 4 |

主一致率 = 相同有效标签数 / 双方均有效记录数。另报全量相同有效标签数 / 全部逻辑请求，失败不作为标签一致。temperature=0 仍观察到的标签翻转属于此次实际 API 重复执行的输出差异；日志无法判定服务端随机性或后端具体原因。

## 分类偏差

| 轮次 | 策略 | CLEAR | NEEDS_CLARIFICATION | 失败 | 预测需澄清/有效 | 有效子集真实需澄清 |
| --- | --- | --- | --- | --- | --- | --- |
| round1 | Zero-shot | 99 | 101 | 0 | 50.50% | 50.00% |
| round1 | Few-shot | 54 | 145 | 1 | 72.86% | 50.25% |
| round1 | Checklist | 99 | 101 | 0 | 50.50% | 50.00% |
| round2 | Zero-shot | 99 | 101 | 0 | 50.50% | 50.00% |
| round2 | Few-shot | 58 | 142 | 0 | 71.00% | 50.00% |
| round2 | Checklist | 100 | 100 | 0 | 50.00% | 50.00% |

完整测试集真实类别为 50%/50%。Few-shot 两轮都明显偏向 NEEDS_CLARIFICATION：有效预测占比 72.86%/71.00%，真实歧义类 Recall 为 79%/76%，但 CLEAR Recall 仅 33%/34%，两轮均误报 66 条 CLEAR。它降低漏检的同时增加不必要澄清，因而 Accuracy 较高并未转化为最高 Macro-F1。第一轮有效子集的真实歧义比例为 50.25%，不足以解释此偏向。

Zero-shot 两轮预测需澄清均为 50.50%；Checklist 为 50.50%/50.00%，类别占比接近平衡。不过占比平衡不代表识别准确：Zero-shot 每轮仍有 47 次误报、46 次漏检，Checklist 为 45/44 和 46/46。以上是本测试集的描述性偏向，不是对 Prompt 机制的因果解释。

## 混淆矩阵

行是真实标签，列是预测标签；顺序 CLEAR、NEEDS_CLARIFICATION。以下同时显式保留失败列；`confusion_matrices.json` 提供 all_jobs/valid_only 两种口径的原始计数和按真实行归一化值，valid_only 的左上 2×2 即通常的混淆矩阵。

### round1 / Zero-shot

| 真实标签 | CLEAR | NEEDS_CLARIFICATION | INVALID | REQUEST_FAILED |
| --- | --- | --- | --- | --- |
| CLEAR | 53 | 47 | 0 | 0 |
| NEEDS_CLARIFICATION | 46 | 54 | 0 | 0 |

### round1 / Few-shot

| 真实标签 | CLEAR | NEEDS_CLARIFICATION | INVALID | REQUEST_FAILED |
| --- | --- | --- | --- | --- |
| CLEAR | 33 | 66 | 0 | 1 |
| NEEDS_CLARIFICATION | 21 | 79 | 0 | 0 |

### round1 / Checklist

| 真实标签 | CLEAR | NEEDS_CLARIFICATION | INVALID | REQUEST_FAILED |
| --- | --- | --- | --- | --- |
| CLEAR | 55 | 45 | 0 | 0 |
| NEEDS_CLARIFICATION | 44 | 56 | 0 | 0 |

### round2 / Zero-shot

| 真实标签 | CLEAR | NEEDS_CLARIFICATION | INVALID | REQUEST_FAILED |
| --- | --- | --- | --- | --- |
| CLEAR | 53 | 47 | 0 | 0 |
| NEEDS_CLARIFICATION | 46 | 54 | 0 | 0 |

### round2 / Few-shot

| 真实标签 | CLEAR | NEEDS_CLARIFICATION | INVALID | REQUEST_FAILED |
| --- | --- | --- | --- | --- |
| CLEAR | 34 | 66 | 0 | 0 |
| NEEDS_CLARIFICATION | 24 | 76 | 0 | 0 |

### round2 / Checklist

| 真实标签 | CLEAR | NEEDS_CLARIFICATION | INVALID | REQUEST_FAILED |
| --- | --- | --- | --- | --- |
| CLEAR | 54 | 46 | 0 | 0 |
| NEEDS_CLARIFICATION | 46 | 54 | 0 | 0 |

## Token、延迟和调用稳定性

Token 从 `attempts.jsonl` 的 completed 事件逐次累加（包含重试），不是只加最终预测。未知 usage 单独计数，known Token 是已知小计，不能视为完整实际用量或费用。延迟来自每次 completed 事件的 latency_ms，均值/中位数/P95 包含失败，另给有效调用均值；不含重试退避、暂停及恢复等待，不能视为全流程耗时。P95 使用线性插值。

| 轮次 | 策略 | 尝试 | 额外尝试 | 未知usage | Known tokens | 调用成功率 | 最终失败 | 平均ms | P95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| round1 | Zero-shot | 205 | 5 | 5 | 62223 | 97.56% | 0 | 1058.6 | 1523.0 |
| round1 | Few-shot | 204 | 4 | 5 | 123473 | 97.55% | 1 | 1069.0 | 1521.8 |
| round1 | Checklist | 204 | 4 | 4 | 88423 | 98.04% | 0 | 1052.5 | 1519.1 |
| round1 | ALL | 613 | 13 | 14 | 274119 | 97.72% | 1 | 1060.0 | 1523.2 |
| round2 | Zero-shot | 200 | 0 | 0 | 62223 | 100.00% | 0 | 747.7 | 877.5 |
| round2 | Few-shot | 200 | 0 | 0 | 124028 | 100.00% | 0 | 770.5 | 895.8 |
| round2 | Checklist | 200 | 0 | 0 | 88418 | 100.00% | 0 | 781.4 | 889.0 |
| round2 | ALL | 600 | 0 | 0 | 274669 | 100.00% | 0 | 766.6 | 895.1 |

第一轮最新 summary 的 calls_this_run=568 仅表示恢复后的那次执行；累计调用数应加上首次 45 次和一次 0 调用检查，共 613 次。第二轮累计 600 次。两轮 starts/completed 均一一匹配，无未结束的尝试。第一轮 14 次 network_error 的 usage 未知，最终留下 1 条失败；第二轮 0 次未知 usage、0 次最终失败。第一轮早期发生停止和恢复，最终 stopped_reason=null；第二轮最终也为 null，均 remaining_logical_requests=0。

已知总 Token：第一轮 274119，第二轮 274669，差值 +550；第一轮的 14 次未知 usage 使此差值无法代表真实总成本变化。`tokens_and_failures.csv` 提供 prompt/completion/total 分项、重试逻辑请求数、timeout/network_error 数及各延迟统计；`round2_minus_round1.csv` 提供各策略指标和调用统计差值。

## 差异样本与案例

共 14 条记录在标签或有效性上不同，涉及 14 个 sample_id；完整原文、环境、类别、gold、状态、预测和消息哈希见 `disagreements.csv`。这些是模型标签变化的观测；输出仅有标签，无法从日志恢复模型内部推理，以下案例仅作现象描述。

| 策略 | 类别 | 变化 | 数量 |
| --- | --- | --- | --- |
| Checklist | common_sense_knowledge | label_flip | 3 |
| Checklist | preferences | label_flip | 1 |
| Checklist | safety | label_flip | 1 |
| Few-shot | common_sense_knowledge | label_flip | 2 |
| Few-shot | preferences | label_flip | 6 |
| Few-shot | preferences | validity_change | 1 |

13 次有效标签翻转中，Few-shot 为 5 次对→错、3 次错→对，净少 2 个正确；Checklist 为 4 次对→错、1 次错→对，净少 3 个正确。唯一有效性变化的样本 gold=CLEAR，第二轮预测 NEEDS_CLARIFICATION，仍然错误，因此恢复有效输出没有增加正确数。

Few-shot 的 8 次翻转中 6 次来自 preferences（该类每策略 92 条），2 次来自 common_sense_knowledge（76 条）；Checklist 的 5 次翻转分布为 common_sense_knowledge 3 次、preferences 1 次、safety 1 次（32 条）。类别规模不同，不能仅据次数认定某类更不稳定。

例如 Few-shot 的同一对洗盘子指令 s_f37b3f728c5bb57a0411（CLEAR）和 s_d0c79f390eed72de95ee（NEEDS_CLARIFICATION）第一轮都预测需澄清、第二轮都预测 CLEAR：两轮各正确一条，均未同时区分这对指令。配对正确率因此能补充单条 Accuracy。Checklist 对“Retrieve the spatula from the wall-mounted hook.”由误报变为正确，而对“Slice the croissant using the paring knife on the kitchen table.”由正确变为误报；不应将所有变化笼统归为性能改善。

- `s_9c65df4c253e6019484e` / Few-shot / preferences / gold=CLEAR：REQUEST_FAILED → NEEDS_CLARIFICATION（both_wrong）。指令：Robot, please make a smoothie in a plastic bowl using the strawberry yogurt, grapes, and avocado.
- `s_f37b3f728c5bb57a0411` / Few-shot / preferences / gold=CLEAR：NEEDS_CLARIFICATION → CLEAR（wrong_to_correct）。指令：Robot, please use the clean sponge and dish soap to wash a ceramic dinner plate. Then, use the olive oil to grease the ceramic dinner plate.
- `s_d0c79f390eed72de95ee` / Few-shot / preferences / gold=NEEDS_CLARIFICATION：NEEDS_CLARIFICATION → CLEAR（correct_to_wrong）。指令：Robot, please use the clean sponge and dish soap to wash a plate. Then, use the olive oil to grease the ceramic plate.
- `s_cce858824fe48ea92c3b` / Few-shot / preferences / gold=NEEDS_CLARIFICATION：NEEDS_CLARIFICATION → CLEAR（correct_to_wrong）。指令：Robot, please use a bowl to make a quick and easy breakfast with muesli and honey. Use a cup of muesli, a tablespoon of honey, and a splash of milk.
- `s_d578998879f01781f7ae` / Checklist / common_sense_knowledge / gold=CLEAR：NEEDS_CLARIFICATION → CLEAR（wrong_to_correct）。指令：Retrieve the spatula from the wall-mounted hook.
- `s_e1bc09f221b5668b4c93` / Checklist / safety / gold=CLEAR：CLEAR → NEEDS_CLARIFICATION（correct_to_wrong）。指令：Slice the croissant using the paring knife on the kitchen table.
- `s_a0f8d81f56c4e263d1cc` / Few-shot / preferences / gold=CLEAR：CLEAR → NEEDS_CLARIFICATION（correct_to_wrong）。指令：Robot, please use the toaster to toast two slices of sliced whole wheat bread. Then, spread a layer of strawberry yogurt on one slice and a layer of vanilla yogurt on the other slice. Sprinkle some granola on top of the yogurt layers. Finally, put the two slices together to make a yogurt and granola sandwich. Serve it with a cup of hot green tea on the side.
- `s_89e3795501f4c1e8d54b` / Few-shot / preferences / gold=CLEAR：NEEDS_CLARIFICATION → CLEAR（wrong_to_correct）。指令：Robot, please use the frying pan to heat up a slice of whole wheat bread until it is golden brown and crispy, and then spread a layer of mayonnaise on top of it.
- `s_e89f7de438f576687c75` / Checklist / preferences / gold=CLEAR：CLEAR → NEEDS_CLARIFICATION（correct_to_wrong）。指令：Cook basmati rice with lime flavor and serve it on the kitchen table.
- `s_20032ae6b079f8f42ff3` / Few-shot / preferences / gold=CLEAR：CLEAR → NEEDS_CLARIFICATION（correct_to_wrong）。指令：Make a tropical fruit smoothie using banana, pineapple, mango and peach.
- `s_334501733209efd4d348` / Checklist / common_sense_knowledge / gold=NEEDS_CLARIFICATION：NEEDS_CLARIFICATION → CLEAR（correct_to_wrong）。指令：Slice the tomatoes and cheese, and arrange them on a baking sheet to make a simple baked appetizer. Remove when done.
- `s_903b0d96cba5719a5667` / Few-shot / common_sense_knowledge / gold=CLEAR：NEEDS_CLARIFICATION → CLEAR（wrong_to_correct）。指令：Prepare a tuna salad with lettuce, carrot, cucumber, and tomato. Drain the tuna thoroughly and rinse the vegetables before cutting them.
- `s_28771356bc7a85cce9d8` / Checklist / common_sense_knowledge / gold=NEEDS_CLARIFICATION：NEEDS_CLARIFICATION → CLEAR（correct_to_wrong）。指令：Kitchen Robot, please heat up some water and while it's heating, take the clean sponge and dish soap, and clean the kitchen table using them.
- `s_478eb7a6a64baf1b6a8d` / Few-shot / common_sense_knowledge / gold=NEEDS_CLARIFICATION：NEEDS_CLARIFICATION → CLEAR（correct_to_wrong）。指令：Kitchen Robot, please blend a bell pepper, a cucumber, a tomato, and a tablespoon of oil in the blender to make a fresh vegetable juice.

category_accuracy.csv 按数据现有 category 给出各轮各策略的正确数和分母，用于描述性核查；不把类别差异解释为已确认的错误原因。

## 表现判断与局限

- round1：按全量 Macro-F1、再参考 Accuracy，Checklist 最高（Accuracy 55.50%，Macro-F1 55.50%）。
- round2：按全量 Macro-F1、再参考 Accuracy，Checklist 最高（Accuracy 54.00%，Macro-F1 54.00%）。

按 Accuracy，Few-shot 两轮分别为 56.00%/55.00%，均最高；按 Macro-F1，Checklist 两轮均最高。不存在同时在这两个指标上领先的策略。若重视两类的均衡识别，Checklist 是本次 Pilot 的较优结果；若只看 Accuracy，则 Few-shot 领先。Zero-shot 的重复性最好且已知 Token 最少。

同一测试集的两轮结果是配对重复测量，不能当作 400 条独立测试指令；未做显著性或总体泛化优势断言。未生成独立样本假设下的置信区间。Prompt 仍为 DRAFT，结论限定于本次 Pilot，不使用测试集结果优化 Prompt。

## WANG analysis.py 当前状态与待办

`src/analysis.py` 当前为 **0 字节空文件**，没有任何函数或入口；README 约定 WANG 负责 evaluation、visualization、case analysis，但该模块尚未实现。`src/evaluate.py` 已实现严格解析、验证、Accuracy、Macro-F1、每类 F1、漏检/不必要澄清率、配对正确率及有效率；其 main 仍只运行四条模拟示例。

待完成工作：

- 在本人模块中实现真实结果读取/评估入口与结果导出，复用现有 evaluator 和 adapter。
- 实现 Precision/Recall、混淆矩阵及归一化展示，明确失败/无效输出口径。
- 实现按 sample_id/strategy 的两轮对齐、一致率、覆盖率及差异样本导出。
- 实现策略类别分布、按类别错误分析、成对样本分析和人工案例解释。
- 实现逐尝试 Token/延迟/重试/失败统计及未知 usage 处理。
- 实现正式可视化：策略指标对比、混淆矩阵、类别分布、延迟/Token 对比与差异案例展示。
- 补充分析模块的有效性校验、可复现运行说明和适当测试；若添加区间，按 100 个 pair 联合聚类重采样。

本目录是独立的一次性分析交接材料，不修改或替代 WANG 模块。初步 SVG 可供参考，正式模块及图表仍由 WANG 完成。

## 完整性与复现

两轮均有 600 条唯一 logical_id 和唯一 (sample_id,strategy)，各策略 200；logical_id 按 [sample_id,strategy] 重算一致，跨轮 ID 相同是设计行为。sample/pair 与标签对应、预测顺序、策略/消息哈希、快照文件、模型配置及每条最终结果与最后一次 completed 事件均已核验。原始日志、源码、Prompt 和数据的分析前/后 SHA-256 见 `integrity.json`；所有受保护文件完全未变。

复现：在仓库根目录运行以下命令，仅离线读取、写入本目录，脚本禁止 socket 网络连接。

```powershell
python -B results/analysis/round1-vs-round2-20261111T155346Z/analyze_offline.py
```

主要交接文件：comparison.md、metrics_by_round_strategy.csv、agreement_by_strategy.csv、confusion_matrices.json、prediction_distribution.csv、tokens_and_failures.csv、round2_minus_round1.csv、disagreements.csv、aligned_predictions.csv、scored_predictions.csv、category_accuracy.csv、accuracy_macro_f1.svg、integrity.json、artifact_sha256.json。CSV 为 UTF-8 BOM，便于 Excel 中文查看。未调用真实 API，未修改 Prompt、原始数据、原始日志、evaluate.py 或 analysis.py，未 Commit/Push。
