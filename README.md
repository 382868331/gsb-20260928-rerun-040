# 生化反应实验系统（第一版）

这是完全离线的小规模人工反应系统，不连接实验设备，也不绘制预设曲线。随机实验由反应网络和质量作用规则驱动；确定性计算读取同一个模型文件。

## 结构

- `biochem/models.py`：物种、反应、初始数量和人工守恒网络。
- `biochem/rules.py`：质量作用倾向与连续质量作用导数。
- `biochem/stochastic.py`：Gillespie 随机运行器、逐次变化原因、时间和可恢复状态。
- `biochem/deterministic.py`：同一模型的 RK4 确定性计算。
- `biochem/analysis.py`：守恒检查和多次短实验的平均变化比较。
- `biochem/persistence.py`：模型与中途轨迹的 JSON 保存/读取。

内置网络为 `E + S ↔ ES → E + P`。它有两条守恒关系：`E + ES` 守恒，`S + ES + P` 守恒。任一反应物数量不足时，对应倾向为零，反应不会发生。

## 使用

```bash
python -m biochem.cli init --model artifacts/model.json
python -m biochem.cli run --model artifacts/model.json --state artifacts/t1.json --end-time 2
python -m biochem.cli run --model artifacts/model.json --state artifacts/t1.json --end-time 5
python -m biochem.cli experiments --model artifacts/model.json --runs 50 --duration 5
python -m biochem.cli deterministic --model artifacts/model.json --end-time 5 --step-size 0.05
```

第一次 `run` 使用种子生成轨迹并保存随机数状态；第二次直接读取同一个状态继续。保存后重放或继续都会沿同一条随机轨迹，而不是只复用初始种子。每个事件都记录反应编号、发生时间、等待时间、被选倾向、总倾向、数量变化、选择原因和变化后数量。

## 中途保存与继续

保存可能发生在下一次反应到来之前：已抽出的等待时间和反应会作为 `pending_event` 与推进后的随机数状态一起写入 JSON，恢复时直接沿用，绝不重新抽取。多次分段、恰好落在反应边界、连续运行三种方式在同一 seed 下产生完全相同的事件序列。

## 指定时刻改变反应条件

```bash
python -m biochem.cli run --model artifacts/model.json --state artifacts/t1.json --end-time 5 --condition 2.0:catalyze=3.0
```

`--condition TIME:REACTION=RATE` 可重复指定。变更会作为可重放输入记录在状态的 `condition_schedule` 中，生效后写入 `condition_events`；保存/恢复后按计划继续生效，分段与连续运行结果一致。若下一次抽取的事件会跨过变更时刻，该次试探性抽取被丢弃，时间推进到变更点后用新条件重新抽取。

待处理事件（`pending_event`）与条件生效时刻严格按时间排序处理：若在条件生效前保存、而待处理事件落在条件生效之后，恢复时会回滚这次抽取（`pending_event` 同时保存抽取前后的 RNG 状态），把时间推进到条件点并用新条件重抽，与连续运行完全一致；事件恰好与条件同时刻时仍归属于生效前那次抽取。因此在条件生效前、恰好生效时、紧随其后保存并多次重开，都得到同一条事件序列。

每个不同的条件生效时刻引入一个单调递增的条件版本：初始为 0，同一时刻的多条变更共享同一个版本号。每个反应事件记录其所属的 `condition_version`，`condition_events` 中也记录对应版本，便于说明事件采用的是哪个条件版本。

## 从同一检查点比较不同条件方案

```bash
python -m biochem.cli compare --model artifacts/model.json --seed 3 \
    --checkpoint 1.0 --end-time 4 \
    --scenario fast:catalyze=5.0 \
    --scenario slow:catalyze=0.01 \
    --state-dir artifacts/scenarios
```

`compare` 从同一检查点深拷贝出多个分支（`StochasticRunner.branch` / `TrajectoryState.clone`），每个分支拥有独立的 RNG 状态，方案之间不会共享后续随机抽取；检查点本身不被修改。各分支在检查点时刻应用各自的速率方案后独立推进，并输出守恒检查、事件数和条件版本。

## 异常停止

运行中发生异常时，状态标记为 `status="failed"`，`failure_reason` 记录失败原因，数量与事件序列保持在最后一次可信状态；CLI 仍会保存状态文件并以非零码退出。
