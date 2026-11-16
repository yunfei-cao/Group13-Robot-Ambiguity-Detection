# v0.1 / v0.3 Windows 开发对照

本次仅增加独立适配器、开发对照脚本和测试。CAI 的 `prompt_v0_3/`、旧 `prompts/` 与 `src/prompts.py`，WANG 的评分源文件，以及历史实验快照均未修改。没有真实 API 调用、commit 或 push。

## 精确故障与修复

故障在 `prompt_v0_3/candidate/prompts.py:172`：

```python
if expected != hashlib.sha256(path.read_bytes()).hexdigest():
```

`path` 为本版 `prompts/few_shot_examples.json`；当前 Windows checkout 是 CRLF，原 manifest 的 SHA-256 对应 LF，随后第 173 行抛错。原文件不改，因此直接调用 CAI 原脚本仍保留此行为；经过新适配器的 Windows 运行已通过验收。

新增 `src/prompt_v03_adapter.py` 的 `VerifiedBundle`：

1. 用 CAI 提交 `0a216fc` 的 Git blob SHA-256 固定两版源码与 manifest 的信任锚。
2. 只做字节级 `CRLF -> LF`，不忽略 BOM、空白、单独 CR 或其他内容变化。规范化后必须与固定信任锚完全一致。
3. 依据已验证的 CAI manifest，逐个验证全部五项导出资产；任何不符都在执行 Prompt 模块前失败。
4. 为源码、manifest 和每项资产记录 `raw_sha256`、`lf_sha256`、`expected_lf_sha256` 与 `crlf_count`。
5. 在适配器管理的临时目录写入已验证 LF 副本，按源码加载，不执行仓库 `.pyc`。调用 CAI 原有默认 `load_examples()` 和 `build_messages()`，不显式传示例绕过默认哈希保护，也不替换哈希函数。
6. 同时验证导出的 TXT 与模块生成的 system 文本一致、示例 ID/配对/标签顺序一致。临时副本随 context 关闭清理。

准备目录中的 `baseline/snapshot/hash_audit.json` 和 `candidate/snapshot/hash_audit.json` 保存原始与 LF 哈希；`snapshot/verified_lf/` 是新建的独立资产副本，不是对 CAI 或历史快照的修改。CheckOnly 会重新验证当前原始目录、持久快照、消息和源码封印。

## 数据与运行目录

已准备的对照目录：

```text
results/prompt_comparison/group13-dev-v01-v03-20261113T150521Z-b2a4a074/
  development_overlap.json
  development_pairs.csv
  run_baseline.ps1
  run_candidate.ps1
  baseline/
    plan.json
    preflight.json
    model_config.json
    snapshot/{dev_inputs.csv,messages.json,hash_audit.json,verified_lf/...}
    pilot/group13-dev-v01-v03-20261113T150521Z-b2a4a074-baseline/
  candidate/
    ...
    pilot/group13-dev-v01-v03-20261113T150521Z-b2a4a074-candidate/
```

两个输出目录初始均无 attempts、predictions 或 run_config。Mock 验收及模拟真实入口的 fixture 全在临时目录，未进入上述目录。每版 170 条开发指令 × 三策略 = 510 个逻辑请求，两版合计 1020，重试另计。

示例配对并集 `dev_0, dev_8, dev_12, dev_36, dev_63, dev_77, dev_98` 的全部成员均排除，原开发 92 对剩 85 对。评估输入与原开发记录完全相同，示例与原开发标签一致，规范化指令文本无示例复用。推理消息仅含当前 environment/instruction，加策略正常使用的固定示例标签；类别、当前标签及配对另一条指令不传给模型。

历史运行依据 `results/dev/*/selected_inputs.csv` 与其实际 attempts/predictions 日志判断，而非把仅准备未执行的样本计为运行。85 对中历史使用 **9 对/18 条**，此前未运行 **76 对/152 条**。历史配对为：

```text
dev_1, dev_10, dev_11, dev_13, dev_15, dev_17, dev_18, dev_19, dev_2
```

逐对标记和历史 run_id 在 `development_pairs.csv`（可直接查看）与 `development_overlap.json`。后续评分应分别报告全部 85 对、历史 9 对、此前未运行 76 对；“此前未运行”不代表作者从未查看过这些开发记录。包内 10 对 pilot 是这 85 对的子集，不可重复计数。作者已经看过旧测试结果，旧测试集不属于新版独立盲测。

## 现有保护如何保留

新增 `scripts/prompt_v03_comparison.py` 复用现有 `ModelClient` 和 `run_experiment()`，不修改二者。为支持未冻结开发候选的快照与预算扩额，运行器使用 `split=pilot`；外层 plan 明确 `dataset_split=dev`、`experiment_kind=development_comparison_not_final`，绝不调用旧测试输入。

- 固定模型 `qwen-plus-2025-12-01`，温度 0，输出上限 64，thinking=false，stream=false，超时 30 秒。
- 每版单独锁、run_id、配置与日志；同一命令续跑跳过最终记录，不能混入旧轮次或另一版。
- 沿用 2/4 秒重试延迟、每任务最多三次 attempt，401/403、三次限流、连续失败、异常/缺失 usage 停止，以及单次用量限制。
- 每版初始总 Token 软停止阈值 **300000**。这是累计已知用量的停止阈值，不是预付费硬上限；可能在 510 任务完成前停止。重试和最后一条请求会影响实际用量。
- 原因未解决的权限/失败停止保持锁定；预算停止只有显式向上扩额才可继续。不得借扩额重试历史终态失败来挑选结果。
- CheckOnly 不读取凭据、不构造真实客户端、不修复或截断日志；崩溃尾部仍交现有运行器恢复。代码、资产、输入或历史审计证据发生变化时拒绝继续旧运行，应准备新对照。
- 准备与核验只读开发标签作示例来源核验，不读取测试标签。推理用冻结 messages，评分另行读取开发 labels。不要用旧测试标签优化候选。

## PowerShell 命令

从仓库根目录执行。以下变量指向已经准备好的这一轮，不要重复 prepare 来续跑。

```powershell
Set-Location 'D:\Group13-Robot-Ambiguity-Detection'
$comparison = '.\results\prompt_comparison\group13-dev-v01-v03-20261113T150521Z-b2a4a074'

# 只读检查：本次已实际执行，两版均通过，不发 API。
& "$comparison\run_baseline.ps1" -CheckOnly
& "$comparison\run_candidate.ps1" -CheckOnly
```

以下是真实付费调用命令，**本次未执行**。用户应事先在当前 PowerShell 进程安全配置 `DASHSCOPE_API_KEY`，不要把密钥写入配置、命令记录或仓库。按以下两条分别启动同一完整开发集的两版实验；已完成的 9 对历史开发记录不会作为这轮结果复用。

```powershell
& "$comparison\run_baseline.ps1"
& "$comparison\run_candidate.ps1"
```

重复相同命令会续跑对应版本。若仅因预算达到停止，审阅该版本 `execution_summaries.jsonl` 和 usage 后，例如明确把该版预算提高至 400000：

```powershell
# 仅在确认预算停止并决定扩大预算后执行；可按需要选择对应版本。
& "$comparison\run_candidate.ps1" -TokenBudget 400000 -ExtendTokenBudget
```

新建一轮而非续跑时才使用：

```powershell
python -B -m scripts.prompt_v03_comparison prepare
```

若本机策略阻止 `.ps1`，可用等效 Python 入口，无须改变机器执行策略：

```powershell
python -B -m scripts.prompt_v03_comparison check --plan "$comparison\baseline\plan.json"
python -B -m scripts.prompt_v03_comparison check --plan "$comparison\candidate\plan.json"
# 以下两行才是真实请求入口，本次未执行。
python -B -m scripts.prompt_v03_comparison run --plan "$comparison\baseline\plan.json" --real-api
python -B -m scripts.prompt_v03_comparison run --plan "$comparison\candidate\plan.json" --real-api
```

## 验收证据

`results/analysis/prompt-v03-integration-20261113/` 保存 67 项测试日志和 acceptance.json：原 59 项相关回归 + 新 8 项测试全部通过。未运行会读取测试标签的整套评分集成测试。

新测试包括当前原始 Windows 目录的兼容性、CAI 默认哈希保护仍有效、LF/CRLF 两种 checkout 消息一致、源码/manifest/任意资产内容篡改均拒绝、示例复用拒绝、两版各 510 条 Mock 预测与逐条消息一致、续跑零调用、错误版本/外来日志拒绝、Token 停止/扩额恢复和权限错误停止。原始目录不需要手动改为 LF。

两版持久 PowerShell 启动器也实际通过 `-CheckOnly`。受保护文件 105 项（包含此前 72 项、全部 30 个 CAI bundle 文件及其他原有模型/评分源码）哈希保持不变；历史两轮 600 条结果完整保留。完整受保护文件列表在 protected_sha256.json。所有测试禁止网络及测试 gold 文件访问；真实模型请求为 0。临时 fixture 输出不代表准确率成绩。
