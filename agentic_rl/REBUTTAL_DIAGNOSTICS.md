# Rebuttal 统计与打印说明

新增功能只记录观测，不改变当前奖励、路由、停止门控或训练目标。

## 直接重跑训练

沿用现有启动命令即可，以下设置默认启用：

```yaml
agentic:
  rebuttal_diagnostics: true
  rebuttal_print_every: 1
  rebuttal_save_turns: true
  rebuttal_output_dir: null
```

每步会打印 `[rebuttal:...]` 行，包含分子和分母。无分母时显示 `NA`，不显示为 0%。
`rebuttal_print_every=10` 可减少控制台打印；每个批次的 JSONL 仍完整保存，包括没有产生更新的批次。

默认保存到：

```text
<ckpt_dir>/rebuttal/<UTC时间-随机运行编号>/
  manifest.json
  diagnostics.jsonl
```

请将 `ckpt_dir` 或 `agentic.rebuttal_output_dir` 指向持久化挂载。每次启动产生独立目录，恢复训练不会覆盖之前的诊断。
`manifest.json` 保存完整配置、源文件哈希、Git revision、消融模式和奖励口径。
`diagnostics.jsonl` 保存按步分组统计、紧凑逐 turn 奖励及优势、候选答案、角色事件和评估结果，不保存完整 prompt 或生成正文。

## 已有 checkpoint：只采样，不重新训练

在 `agentic_rl` 目录执行，例如：

```bash
python diagnose_rebuttal.py \
  --checkpoint /持久化目录/已有checkpoint \
  --output /持久化目录/rebuttal-probe \
  --batches 3 \
  --questions_per_batch 32 \
  --samples_per_question 16 \
  --eps_force 0.25
```

默认读取训练集，使用 temperature=1.0 的训练形态采样，不创建 optimizer、不更新参数。
使用 `--config` 提供与所分析模型相符的配置，并核对 `--eps_force`。
`--seed` 控制问题抽样及注入 RNG，不保证 vLLM 生成完全可复现。
这属于新采样的 checkpoint 诊断，不能描述成已经丢失的原训练历史。
如保留了多个训练阶段的 checkpoint，可分别运行并报告阶段变化。

## 导出表格

```bash
python summarize_rebuttal.py /持久化目录/rebuttal-probe/运行编号/diagnostics.jsonl \
  --output /持久化目录/rebuttal-tables --phase checkpoint_probe
```

训练日志使用 `--phase train`；独立评估或训练内评估可以选择 `eval_aime`、`eval_math_l5` 等。
支持传入多个日志，但保留 run_id 和 phase，不把不同 seed 或训练采样与 greedy 评估混为一个结果。
覆盖率由累加后的分子和分母计算，不平均每个 batch 的百分比。

## AIME 等逐题配对分析

```bash
python compare_rebuttal_predictions.py RACA逐题结果.jsonl 基线逐题结果.jsonl \
  --output 配对统计.json
```

打印两方法分别答对数、各自独有答对数、准确率差、双侧精确 McNemar p 值，以及按题目配对 bootstrap 的差值区间；包含年份元数据时还会按年输出。
每次比较一个 seed/checkpoint 对；工具拒绝题目集合不同或同一文件中重复的题目，避免把多 seed 的重复题目当作独立样本。
不同 seed 的训练方差仍应独立报告。

## 覆盖的审稿问题

| 问题 | 记录内容 |
|---|---|
| H-W03 / H-Q01 / A-Q02 分组稀疏 | role/context/response/channel/p 分解；原始条目数、去重代表数、组大小直方图；空组、代表数不足、低方差分别记录；有效 turn、有效通道及对应分母 |
| H-W04 奖励幅度被归一化消除 | 每组原始奖励分布、均值、总体标准差、n=2 比例；逐 turn 的原始奖励与实际优势；配置中的权重 |
| H-W01 / H-W02 控制器 | 每次提出 stop/continue、是否存在评分、最新黑板评分、是否接受/拦截、轮次和上下文；机械自动停止单独计数 |
| H-W05 / H-Q04 critic 延迟标记 | 按轮次/context/hop 的 TP/FP/TN/FN、terminal flag、可观察修复成功/失败、实际奖励 |
| H-W06 / H-Q05 verifier 折中 | 真实被审答案正确性对应的分数、均值/标准差、0.5 及附近比例、Brier、校准分箱/ECE、解析失败、验证调用来源；低方差过滤另见 anchor 表 |
| H-W07 / A-W04 / A-Q03 / U-W05 / U-W07 选择性 | context/p/target/forced 分解的请求数、轮末正确、修正、破坏、失败请求及奖励 |
| H-W08 / H-Q02 / U-W06 修复和投票 | 初始答案、最后候选、最后 primary、全候选等权投票和生产答案的正确性；episode 级 fix/break 与逐次 correction 变化分开记录 |
| H-W09 / H-W10 / A-W02 / A-Q04 / U-Q01 统计分析 | 评估 JSONL 保留逐题预测、gold、题目、年份/来源和各种答案选择结果，供跨方法逐题配对及跨 seed 分析 |
| H-W13 / A-W05 / U-W01 / U-Q02 成本 | 各角色调用数、实际 prompt 输入 token、输出 token；每批 rollout/update 时间及实际执行配置 |

## 统计口径

- `anchors.csv`：通道覆盖率 = 通过组大小和方差筛选的原始条目数 / 该通道所有符合资格的原始条目数。代表样本去重只改变组统计，不减少被代表的 turn 数。
- 同一张表的 `structural_assigned/eligible` 再检查该通道的 token/logprob/span 可用性；一个 turn 只有 solution span 时，不会把它算作可路由的 interaction credit。
- forced primary 不进入 interaction 通道分母；solution 仍按其自身分组规则计算。零权重关闭的通道也不属于可训练 interaction 条目。
- `absent` 表示没有符合资格的条目，可能是未到达该上下文，也可能是对应轮次均不符合通道资格；不要与有样本但低方差混用。
- `turns.csv`：`assigned/total` 统计至少获得一个有效优势的轮次；优势恰好为零仍然是有效分配。`structural_valid/total` 再应用 token/logprob/span 检查，不代表所有轮次都通过了后续数值安全检查。
- 训练 credit 表的分母是进入优势估计的题目组；JSONL 同时保存 requested/returned/admitted episode 数，避免漏掉执行或入组失败。
- `groups=32/32` 是题目组统计，不是细粒度 anchor 保留率。
- verifier Brier 和校准按该 verifier 实际审查的答案计分，不拿最终系统答案作标签。控制器事件同时记录最新黑板评分及 prompt 中展示的最高平均评分，分别按 `score_bin`、`displayed_score_bin` 分箱。黑板评分可能包括原执行器对解析失败的 0.5 兜底；校准统计使用原始解析分数，并排除解析失败。
- 最后候选答案包含 correction，最后 primary 不包含 correction。相同轨迹上改变聚合方式是答案选择诊断，不等于重新训练无投票系统，也不证明角色的因果贡献。
- 新日志的计数每批计算一次，不因 PPO epochs 重复累计。optimizer 原有 channel_count 指标另存为 optimizer_metrics，口径不同。

## 重跑前核对配置

当前代码的 verifier 训练奖励为 `1-abs(v-p)`；论文描述为 `1-(v-p)^2`。本次埋点没有替换训练奖励。
当前配置还可能启用 `verify_after_correction`、`verified_correction_gate`、`auto_stop_verified`、交互分层平衡和按角色归一化等选项。
本次记录会保留实际配置；用于 rebuttal 时，应先确认所跑配置确实对应要分析的论文版本。
当前 `train.py` 还会周期性评估配置中的 MATH test_path 和 AIME 数据；本次没有改变已有评估安排。若要遵守论文的 test-once 协议，需要核对独立验证集与最终测试的配置和运行方式。

仅增加统计不能替代其他模型/任务、预算匹配基线、固定停止策略或角色因果控制实验。跨 seed 结论仍需独立训练运行；预训练污染也不能用这些日志排除。
