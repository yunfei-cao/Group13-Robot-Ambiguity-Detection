# 提示词 v0.3 开发候选版

这是在看到原实验总体结果和部分案例后制作的后续开发候选版，尚未调用模型测试，不能声称已经提高准确率。原数据和标签均未修改。请保留原实验作为第一次实验结果。

## 本次具体怎么改

### 三种策略共享的任务说明

旧版已提醒不要因普通细节缺失而澄清。候选版把判据写得更具体：

- 先由完整上下文确定用户目标和对象；能定位对象的简写指代不自动算歧义。
- 多个食材或直接操作目标都符合要求，且不同选择会改变用户意图时，应澄清。
- 目的地或必需先行信息没有给出时，应澄清。
- 普通工具选择、寻找已命名物品、清洁和防护步骤可由机器人安排，不要求逐项向用户询问。
- 允许规划普通步骤不等于允许猜食材、成品要求、关键前提或用户偏好。

新说明用于同一版本的zero-shot、few-shot和checklist三种策略。checklist进一步按目标、指代、实质影响、最终决定四项组织。仍只输出CLEAR或NEEDS_CLARIFICATION。

### 四个few-shot示例

| 顺序 | 开发配对 | 原标签 | 示例说明 |
|---|---|---|---|
| 1 | dev_77 | CLEAR | 两个鸡蛋和切达奶酪做蛋饼；目标和原料明确，不必列全工具和步骤。 |
| 2 | dev_8 | NEEDS_CLARIFICATION | 环境有多种面包，仅称一片面包，种类未定。 |
| 3 | dev_98 | CLEAR | 去冰箱寻找指定的果酱和炼乳；寻找目标明确。 |
| 4 | dev_12 | NEEDS_CLARIFICATION | 将鸡蛋放到specified location，但没有提供那个位置。 |

仍为四个不同开发配对、两个标签各两条。示例全文均直接来自原开发输入，标签直接来自原开发标签；没有把AI建议改标的记录当作示例。选择理由是人工可读的元数据，不会进入模型messages。来源类别不等于验证了该类别的安全推理能力。

## 包里有什么

- baseline/：旧版提示词模块及原来六个提示词资产。baseline的五个模型文本/示例/预览资产已经与当前项目逐字节核对一致；manifest新增的校验元数据不影响模型文本。
- candidate/：新版prompts.py及生成好的三套TXT、四个示例JSON、预览和版本清单。
- pilot_10pairs/：固定随机种子42抽出的10对开发样本，共20条，两个标签各10条。
- development_comparison_85pairs/：排除两版示例配对并集后的全部共同开发样本，85对、170条，两个标签各85条。
- comparison_manifest.json：示例ID、排除配对、来源哈希及验证记录。

两版共用了7个不同示例配对。因此不是只从开发集排除新版4对，而是同时排除旧版和新版用过的全部7对。这能避免旧版或新版因见过示例答案而在开发比较中获得优势。

## 你怎么交给曹和王

1. 将整个prompt_v0_3目录保存在项目experiments/prompt_v0_3，或把此ZIP发给曹。不要先覆盖原src/prompts.py与原prompts目录。
2. 曹使用baseline和candidate各自的build_messages(strategy, environment, instruction)，分别对同一组pilot_10pairs/dev_inputs.csv运行三种策略。模型、温度、token限制、错误处理保持一致。两版各60个逻辑请求，合计120个；本次没有实际发起API请求。
3. 两版结果分开保存，并记录prompt_version和sample_id。王使用pilot_10pairs/dev_labels.csv的原标签评分，比较Accuracy、Macro-F1、清晰误判率、歧义漏检率及有效输出率。
4. 10对只用于检查调用和观察错误模式，小幅得分变化不能证明改善。若候选版值得继续验证，可在共同85对开发样本上比较；重复使用同一开发集进行选择会使开发得分偏乐观，需要如实记录。
5. 确定候选配置后再安排后续测试。原测试结果已被查看，原测试集上的新版运行应标记为后续实验，不能包装成未见过的独立验证。

旧版与新版同时改变了共享任务说明和few-shot示例，因此比较能回答“这个整体候选配置是否更好”，不能独立证明是换示例还是改说明造成差异。

## 调用注意

仅上传TXT不会自动改变调用原prompts.py的实验脚本。曹需要确认程序确实加载了对应版本的模块，或者明确使用该版本导出的文本。

推荐用文件路径分别加载，避免两个同名prompts模块被Python缓存混淆：

```python
import importlib.util
from pathlib import Path

def load_prompt_module(script_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, Path(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

old = load_prompt_module("experiments/prompt_v0_3/baseline/prompts.py", "prompts_baseline")
new = load_prompt_module("experiments/prompt_v0_3/candidate/prompts.py", "prompts_candidate")
# sample仅来自dev_inputs.csv；两版使用完全相同的环境和指令。
# messages = new.build_messages("few_shot", sample["environment"], sample["instruction"])
```

接口与旧版一致。各模块默认从自身目录的prompts/few_shot_examples.json读示例，不要只移动模块而漏掉配套prompts目录。不要将标签CSV、类别、审核理由或配对的另一条指令传入模型。

本包已完成格式、来源、标签数量、示例隔离、三种策略输入构造和哈希核对；没有做模型效果验证，不保证正确率上升。

## v0.3 新增的直接使用工具

这版进一步限定了两条规则：不为未要求的后续任务虚构目的地；用户明确允许任选合适对象、任一选项或全部对象时，应尊重这种授权。四个示例保持v0.2所选的原开发记录，没有反复更换示例来猜测效果。

候选模块默认加载示例时核对版本及示例文件SHA-256，防止新模块搭配旧示例，或示例被编辑后仍沿用旧版本记录。程序中的三种策略仍共享同一任务说明。

### 曹：读取已经准备好的请求

requests/baseline.jsonl与requests/candidate.jsonl已包含同一组20条开发指令、每条三种策略，共各60个任务。每行是一个JSON对象：sample_id、pair_id、strategy、prompt_version、messages。

曹的API程序读取每行后，将其中messages传给原模型接口，保留其余字段用于记录。不要把整行JSON作为模型输入。模型版本、温度、输出token限制及重试规则保持一致。这里没有保存API密钥，也没有调用API。尚未检查曹实际运行脚本，可能需要增加读取此请求格式的适配。

API结果各保存成一个CSV，至少包括这些列：

```text
sample_id,strategy,prediction,prompt_version
```

strategy只能为zero_shot、few_shot、checklist。旧版prompt_version为v0.1-dev，新版为v0.3-candidate。prediction写实际模型输出；请求最终失败时仍保留这一行，prediction留空。耗时、用量等可继续作为额外列保留。每个CSV应有60条记录。不同运行轮次分别存文件，禁止混在一个CSV里。

如要对共同85对生成请求，可在本目录运行：

```powershell
python experiment_tools.py prepare --inputs development_comparison_85pairs/dev_inputs.csv --variant baseline --out requests/baseline_85pairs.jsonl
python experiment_tools.py prepare --inputs development_comparison_85pairs/dev_inputs.csv --variant candidate --out requests/candidate_85pairs.jsonl
```

prepare只准备消息，不发送请求。不要先运行这一步扩大调用量；先完成10对小规模试跑。

### 王：用同一个评分程序比较两版

假设曹将实际结果存为results/baseline.csv和results/candidate.csv，在本目录运行：

```powershell
python experiment_tools.py compare --labels pilot_10pairs/dev_labels.csv --baseline results/baseline.csv --candidate results/candidate.csv --out results/comparison.csv
```

输出六行对比结果，包含Accuracy、Macro-F1、漏检歧义率、不必要澄清率、有效输出率、配对正确率、无效输出数和多数类基线。

工具要求两版都覆盖标签文件的全部sample_id和三种策略。缺行、多行、重复、混用prompt_version会报错，不会静默丢掉失败样本。不同实验轮次需分别比较。

计分约定：prediction去除首尾空白后必须精确等于两个允许标签之一。其他文本或空值算无效；无效输出在Accuracy和配对正确率中算错，并计入对应真实类别的F1漏检项。漏检歧义率只计算NEEDS被有效判为CLEAR的数量除以真实NEEDS总数，不必要澄清率同理；无效输出单独报告。这一约定应与报告中声明的口径保持一致，不能无说明地覆盖旧报告数值。所有比例以0到1保存。

validation_results.json列出离线核验，包括全对预测、恒定类别、全部无效输出、相同标签配对，以及错版本、缺行和重复行的拒绝检查。这些测试数据仅存于内部验证目录，包中没有伪造的模型实验成绩。
