# Group 13 开发集验收与正式实验准备

开发集日志与评估接口验收通过，但分类覆盖率为59/60，尚有1条结果未知；正式实验当前禁止启动：Prompt仍为DRAFT。

来源：`results/dev/group13-dev-qwen20-20261111T060727Z-d9300025`。标准答案：`data/processed/dev_labels.csv`。仅评估已选20条指令/10个完整配对。

|策略|逻辑数|有效/覆盖率|启动/完成事件|超时尝试|已知输入/输出/总Token|已知尝试耗时(s)|成功平均耗时(s)|
|---|---:|---:|---:|---:|---:|---:|---:|
|zero_shot|20|19/20 (95%)|22/21|2|6002/89/6091|78.435|1.884|
|few_shot|20|20/20 (100%)|20/20|0|12403/105/12508|47.085|2.354|
|checklist|20|20/20 (100%)|20/20|0|8863/95/8958|39.555|1.978|

最终状态：zero_shot为19 success + 1 interrupted；其余各20 success。没有最终error或invalid_output。两次timeout均在重试后成功，全部属于zero_shot，且不是目标interrupted请求。

已知总用量27557 Token。两次timeout和一次interrupted用量未知，不能按零计费；统计为下界，无法仅凭本地日志确认账单总额。已完成尝试耗时合计165.076秒；首末事件跨度230.268秒，包含重启/等待，不是纯模型推理时间。

|策略|有效样本正确数|有效样本Accuracy|有效样本Macro-F1|全部20条Accuracy|全部20条Macro-F1|完整配对正确率|
|---|---:|---:|---:|---:|---:|---:|
|zero_shot|10/19|52.63%|50.43%|50.00%|49.17%|20.00%|
|few_shot|11/20|55.00%|48.72%|55.00%|48.72%|10.00%|
|checklist|11/20|55.00%|52.00%|55.00%|52.00%|20.00%|

全量指标调用未修改的WANG score_samples/calculate_metrics：失败计为错误，并计入真实类别FN。有效样本指标仅过滤已经评分的有效行，以原calculate_f1计算两类平均，不重新用配对校验器处理19条不完整子集，也不报告有效子集配对指标。zero_shot有效子集的类别分布详见JSON；不同策略有效分母不同。初步效果基于首10个合格开发配对，不能当作正式测试集效果或随机抽样估计。

异常请求：`9f2c7b2d7194a7947eba7803725a407eaa776a167b9e5f81a28a4d36d71d63ff`，sample `s_f07e8be2c8b536f86e35` / zero_shot。attempts第33行只有started（2026-11-11T06:22:23.006679+00:00，attempt_number=1）；没有completed、原文、标签、Token或服务端确认，不能从已有日志恢复。predictions第16行为interrupted/outcome_unknown。model_id为配置回填，不证明服务端响应。用户确认后台存在真实调用，但无法关联证明此请求已到达或返回。中断具体原因无异常栈可证；不能将其归因为已记录timeout。

第1–33行：17次启动=15个成功逻辑请求+1次超时重试+1次未知请求；第34–123行：45次启动=44个成功逻辑请求+1次超时重试。累计60条终态=15此前成功+44后续成功+1 interrupted。当前续跑会跳过所有最终记录，包括interrupted；临时副本禁网/禁止客户端调用复验证实新增调用0。

单条补跑提案（未实施）：给运行器增加可选strategies参数及CLI --strategies zero_shot，默认保持三策略，预算与run_config按所选策略计算；独立新run只输入目标sample，并使用原模型/提示词哈希。先离线验证仅1个逻辑请求及重复运行0调用。原重试规则仍可能产生最多3次物理尝试。独立reconciliation manifest记录原/新run、logical_id、未知结果及潜在重复收费，评估时明确映射这一条替代结果；保留原日志。当前仅选sample仍会跑三策略，故不建议直接运行。

正式集实测为200条指令、100个完整配对，CLEAR/NEEDS_CLARIFICATION各100；三策略共600个逻辑请求。数据/Prompt manifest的原始字节哈希存在CRLF/LF差异，全部在仅归一化CRLF为LF后匹配；没有修改源文件或manifest。manifest_hash_checks.json逐项保留预期/实际/归一化哈希，正式准备绑定当前实际字节。正式集与开发集sample_id/pair_id零交集，正式指令与原始开发集direct/indirect/ambiguous指令归一化零重复。开发pilot排除固定示例配对。

固定Few-shot为4个开发示例，来自dev_0/dev_8/dev_63/dev_36，两类各2，顺序和内容与manifest/开发源一致。三份模板与运行时生成文本一致。全部600个计划消息检查通过：当前输入只有environment/instruction，未带gold/category/pair元数据或配对另一条指令；Few-shot演示标签为设计内的开发标签。WANG标签仅在离线评估连接。

模型配置：qwen-plus-2025-12-01，temperature=0，max_tokens=64，timeout=30秒；客户端请求固定enable_thinking=false、stream=false。全部离线测试70项通过，失败0，错误0。

安全机制核对：401/403立即停止；累计3次429或连续3次不成功停止；成功/无效输出缺失或非法用量停止；单次prompt>16000、completion>64、total>16064及累计已知Token>=300000停止。累计阈值在一次尝试完成后检查，属于软停止，可越过阈值；未知请求用量不在累计内。600为逻辑请求上限，不是物理调用硬上限（每条最多3次）。无重试600次时completion上限38400，prompt部分须另计；开发用量直接按10倍缩放仅为275570的粗略下界估算，接近300000，不能保证正式集跑完。

续跑校验输入/配置/Prompt哈希与身份，已有终态跳过；完成事件无最终行可恢复；孤立started仅标记unknown不重发；重复预测/事件及中间损坏拒绝，未换行尾片段会在实际续跑时截断；OS锁阻止并发写入。只保留已确认用量，不能实现跨网络恰好一次调用。客户端没有隐藏自动重试。

正式准备目录：`results/formal/group13-formal-qwen200-prepared-20261111-v1`，独立run_id `group13-formal-qwen200-prepared-20261111-v1`；含model_config.json、preparation_manifest.json、run_command.ps1。没有创建真实run_config或attempts/predictions。准备脚本校验FROZEN/APPROVED状态、版本与输入/模板/示例/代码/模型哈希，当前DRAFT会阻止执行。若冻结涉及版本或内容变化，必须重新准备并复测，不可绕过哈希。正式运行及其后续恢复均须由用户另行授权执行。

现有runner本身不会校验DRAFT状态，正式命令包装增加了显式检查。manifest状态文字必须由Prompt负责人根据实际审核更新；本次不改动或自行冻结。data manifest的manual_semantic_review仍标PENDING，作为已有元数据待办披露，不重新否定用户已接受的200条正式集。

验收判断：离线技术接口条件通过，正式启动条件尚未满足。明确阻塞为Prompt仍DRAFT；开发集完整验收另有1条未知记录需获准补跑或书面接受缺失。单策略补跑能力尚未实现。300000 Token软停止可能提前结束正式批次，届时先审查原因与预算，不可通过更换run_id或删除日志绕过停止。

图表：dev_overview.png/.svg与dev_confusion.png/.svg；逐条评分scored_*.csv；结构化dev_summary.json、strategy_summary.csv、interrupted_diagnosis.json、readiness_checks.json；完整测试输出offline_tests.txt；保护文件哈希source_integrity.json。图表标签用英文确保跨平台渲染。

本次只新增审计脚本、派生报告/图表及准备文件；原始开发日志、CAI数据/Prompt、WANG评估及所有已有src/tests均以哈希验证未改动。未调用真实API、未补跑、未启动正式实验、未Commit/Push、未输出或记录真实API Key。
