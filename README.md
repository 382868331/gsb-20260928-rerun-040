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

## 延迟释放的产物（等待池）

有些反应在“开始”后产物并不能立即进入体系，而要等待一段固定时间。模型在反应上用 `delayed_products` 声明这类产物：

```python
Reaction(
    "catalyze",
    reactants={"ES": 1},
    products={"E": 1},                 # 酶立即释放
    rate_constant=0.20,
    delayed_products={"P": (1, 0.8)},  # 1 个 P 等待 0.8 时间单位
)
```

规则如下：

- 反应开始时立即扣减反应物、加入即时产物；延迟产物不进入自由计数，而是作为**有身份的等待批次**（`WaitingItem`，含全局唯一 `release_id`、数量、创建时刻、计划释放时刻、创建时的条件版本）存入状态的 `waiting`。
- 等待期间这些分子既不能参与任何后续反应，也不影响倾向；它们只出现在等待池里。`analysis.material_balance` 与 `validate_material_balance` 会把“自由分子 + 等待中分子”一起核对，因此产物不可能提前出现或无声消失。
- 到释放时刻生成一条 `ReleaseEvent`（记录批次身份、数量、释放时刻、释放时自由计数与等待池余量、创建版本与释放版本），批次从等待池移除、分子加入自由池。每份批次恰好释放一次，释放 `release_id` 不重复。
- 已经开始等待的反应不受后来条件变更影响：释放时刻、数量、身份在开始时就固定；后来的速率变化只改变尚未发生的随机反应。`ReleaseEvent.created_version` 记录开始时版本，`condition_version` 记录释放时版本。

### 同刻顺序

同一时刻同时发生释放、随机反应与条件切换时，顺序固定为：

1. 该时刻到期的等待批次先释放（按 `release_id` 顺序，即创建顺序），使用该时刻**生效前**的条件版本；
2. 条件切换随即生效，`condition_events` 记录新版本与速率快照；
3. 之后的随机反应在新版本下抽取。

若一次试探性随机事件跨过（或恰好撞上）一个释放时刻，该次抽取连同其随机数一起回滚，先完成释放，再在释放后的状态上重新抽取——这与条件插入时的回滚规则一致。

### 保存点与分支

等待池完整写入 JSON；保存点放在等待中、释放前一瞬、释放当刻或多个释放同刻都可以。运行在恰好结束于释放时刻时会先完成该时刻的全部释放（以及同刻的条件声明）再封顶。恢复后续跑与一次性连续运行逐事件一致（包括 `ReleaseEvent` 序列与随机数状态）。

`fork()` 出的分支各自携带独立的等待池副本与后续随机流：检查点上已存在的等待批次会在各分支中按原定时刻各自释放，身份与时刻一致；分支之间后续新建的批次互不共享，任一分支都不会重复释放同一份产物。可据此比较“改变事件发生时刻”与“只改变尚未完成批次的释放时刻/条件版本”的不同方案——但注意：已开始等待的批次其释放时刻不会被条件变更改动。

内置延迟网络可用 `python -m biochem.cli init --network delayed-release --model artifacts/delayed.json` 生成。

确定性计算（RK4）也读取同一模型：无延迟产物时退化为普通质量作用 RK4；有延迟产物时，每步 RK4 平均通量先进等待队列，到固定延迟后释放。`DeterministicPoint.values` 是自由计数，`DeterministicPoint.waiting` 是该时刻等待中的量，两者相加满足同样的守恒关系。

## 异常停止

运行中发生异常时，状态标记为 `status="failed"`，`failure_reason` 记录失败原因，数量与事件序列保持在最后一次可信状态；CLI 仍会保存状态文件并以非零码退出。
