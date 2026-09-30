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

每条事件还带有 `condition_version`：生成该事件时生效的条件版本号（初始条件为版本 0）。同一时刻声明的多项条件共同构成一个新版本，按声明顺序依次合并；`condition_events` 中每项记录其 `version` 与合并后的完整速率快照 `rates_after`，事件原因字符串也包含所用版本。事件时刻与条件生效时刻恰好相等时，事件先按旧版本触发，条件随即生效。

## 中途保存与继续

保存可能发生在下一次反应到来之前：已抽出的等待时间和反应会作为 `pending_event` 写入 JSON，同时保存抽取前的随机数状态 `rng_state_before`。恢复时若该事件仍然有效则直接沿用；若在保存后、该事件发生前有条件变更插入，则整个试探性抽取连同其消耗的随机数一起回滚丢弃，在条件生效后重新抽取——因此无论保存点落在条件生效前、恰好生效时还是紧随其后，分段、恢复与连续运行都产生同一条事件序列。

## 指定时刻改变反应条件

```bash
python -m biochem.cli run --model artifacts/model.json --state artifacts/t1.json --end-time 5 --condition 2.0:catalyze=3.0
```

`--condition TIME:REACTION=RATE` 可重复指定，同一 `TIME` 可给出多项变更。变更会作为可重放输入记录在状态的 `condition_schedule` 中，生效后写入 `condition_events`（含版本号与速率快照）；保存/恢复后按计划继续生效，分段与连续运行结果一致。若下一次抽取的事件会跨过变更时刻，该次试探性抽取连同随机数一并回滚，时间推进到变更点后用新条件重新抽取。

## 从检查点分出条件方案

需要比较不同条件方案时，使用 `StochasticRunner.fork()`（或 `TrajectoryState.fork()`）从同一检查点深拷贝出独立分支。各分支拥有独立的事件列表、条件计划和随机数状态，互不共享后续随机流；在分支上分别 `schedule_condition_change` 后继续运行即可逐事件对比。

## 异常停止

运行中发生异常时，状态标记为 `status="failed"`，`failure_reason` 记录失败原因，数量与事件序列保持在最后一次可信状态；CLI 仍会保存状态文件并以非零码退出。
