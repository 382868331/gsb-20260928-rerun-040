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

模型可在反应的 `product_delays` 中为产物声明正延迟，例如 `{"P": 1.0}` 表示反应事件发生时先扣减反应物，`P` 作为等待中的中间物记账；到预定时刻才加入 `counts`。每份等待产物都有唯一 `release_id`，记录其来源 `source_event_id`、数量、反应 ID、创建时间、预定释放时间和启动反应时的条件版本。等待中物不进入质量作用倾向，也不会在后续条件改变时消失或被改写。

等待中间物可以有四种去向，状态 JSON 分别以 `counts`（自由体系）、`waiting_counts`（仍等待）、`inactivated_counts`（失活）和 `sequestered_counts`（隔离）计量。部分转移会分裂原挂账，子批次保留 `root_release_id`、`parent_release_id` 和 `source_event_id`，因此每个释放、失活或隔离事件都能追溯回最初启动反应。失活和隔离不是“减少等待量”的账面抵消；守恒检查会把自由、等待、失活和隔离数量一起计入。

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

`--condition TIME:REACTION=RATE` 可重复指定，同一 `TIME` 可给出多项变更。变更会作为可重放输入记录在状态的 `condition_schedule` 中，生效后写入 `condition_events`（含版本号与速率快照）；保存/恢复后按计划继续生效，分段与连续运行结果一致。若下一次抽取的事件会跨过变更时刻，该次试探性抽取连同随机数一并回滚，时间推进到变更点后用新条件重新抽取。已经启动的延迟产物只携带启动时的条件版本；后来改变速率只影响之后新抽取的反应，不改变已挂账产物的数量或释放时刻。释放事件的 `condition_version` 是启动版本，`release_condition_version` 记录释放当下版本。若释放与条件切换同刻，先按稳定的 `release_id` 顺序释放，再生效条件。

## 等待物的竞争去向

等待物竞争事件与条件切换同刻时，固定顺序为：释放闸门切换、等待物转移或显式释放、到期自动释放、反应条件切换、后续反应。也就是说先判断是否允许释放，再执行同刻失活或隔离，最后只释放仍处于等待状态且符合闸门条件的剩余数量；已经显式释放进自由体系的部分不会因之后改变安排而追回。

可调用 `schedule_release_gate(time, allowed)` 暂时阻止或恢复自动释放。阻止不是销毁：到期批次继续保留在等待台账中；恢复时仅仍存在、状态为等待的批次会释放，实际释放事件时间是恢复时间，原定时间保存在 `original_scheduled_time` 以便追溯。隔离批次不会因闸门恢复自动释放，必须先安排一次到 `waiting` 的转移。

可调用 `schedule_waiting_transfer(time, release_id, destination, amount)` 为部分或全部挂账安排去向，`destination` 为 `waiting`、`inactivated`、`sequestered` 或 `free`。同一次源反应的不同部分可分别安排；转移数量不能超过当时仍存在的对应批次，转移后剩余数量允许为零。每个转移都有唯一 `transfer-xxxxxx` 身份；自动释放和显式释放都不会重复使用同一批次。

CLI 可使用 `--release-gate TIME:true|false` 控制释放闸门，使用 `--waiting-transfer TIME:RELEASE_ID=DESTINATION:AMOUNT` 安排竞争去向。闸门事件、转移事件和释放事件都写入普通 `events` 列表；竞争计划本身写入 `release_gate_schedule` 和 `waiting_transfer_schedule`，保存/恢复后按各自当时的条件版本继续。

## 从检查点分出条件方案

需要比较不同条件方案时，使用 `StochasticRunner.fork()`（或 `TrajectoryState.fork()`）从同一检查点深拷贝出独立分支。各分支拥有独立的事件列表、条件计划和随机数状态，互不共享后续随机流；在分支上分别 `schedule_condition_change` 后继续运行即可逐事件对比。若要区分改变事件发生时刻还是尚未完成反应的释放时刻，可对分支调用 `reschedule_pending_release(release_id, scheduled_time)`；它只移动指定挂账，不改变源事件身份、数量或随机轨迹。

状态 JSON 包含 `waiting_counts` 与 `pending_releases`。每个反应事件的 `waiting_after` 给出事件后等待库存，释放事件带 `event_type="release"`、`release_id` 和 `source_event_id`。分支深拷贝后可各自释放检查点上的同一份挂账，但任一分支内同一 `release_id` 只允许出现一次。守恒检查会把各物种的等待数量与同名自由物一并计入。

新增竞争去向后，旧的仅有延迟释放实验无需添加闸门或转移计划：默认闸门始终允许，状态中的新字段自动补默认值，原反应事件、释放事件、随机数轨迹和 `release_id` 身份保持不变。保存点可放在竞争事件之前、当时或之后；从检查点继续会重放相同的闸门、转移、释放和条件切换事件，逐事件库存与连续运行一致。

## 异常停止

运行中发生异常时，状态标记为 `status="failed"`，`failure_reason` 记录失败原因，数量与事件序列保持在最后一次可信状态；CLI 仍会保存状态文件并以非零码退出。
