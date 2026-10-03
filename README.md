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

## 从检查点分出条件方案

需要比较不同条件方案时，使用 `StochasticRunner.fork()`（或 `TrajectoryState.fork()`）从同一检查点深拷贝出独立分支。各分支拥有独立的事件列表、条件计划和随机数状态，互不共享后续随机流；在分支上分别 `schedule_condition_change` 后继续运行即可逐事件对比。若要区分改变事件发生时刻还是尚未完成反应的释放时刻，可对分支调用 `reschedule_pending_release(release_id, scheduled_time)`；它只移动指定挂账，不改变源事件身份、数量或随机轨迹。

状态 JSON 包含 `waiting_counts` 与 `pending_releases`。每个反应事件的 `waiting_after` 给出事件后等待库存，释放事件带 `event_type="release"`、`release_id` 和 `source_event_id`。分支深拷贝后可各自释放检查点上的同一份挂账，但任一分支内同一 `release_id` 只允许出现一次。守恒检查会把各物种的等待数量与同名自由物一并计入。

## 等待中间物的竞争去向

等待中的每份产物除按原安排释放外，还可以通过 `StochasticRunner.schedule_waiting_fate(time, fate_type, ...)` 指定竞争去向。去向各自独立记账，减少等待量不等于反应没有发生过：自由产物在 `counts`，等待物在 `waiting_counts`/`pending_releases`，失活物累计在 `deactivated_counts`，隔离物累计在 `sequestered_counts`/`sequestered_lots`。守恒核对把四个账本一并计入，因此逐事件与总量都可追溯到最初启动反应的 `source_event_id`。

- `inactivate`：对指定 `release_id`（给 `products`/`amount` 可只处理同一份等待物中的部分数量）或全部等待物失活。部分处理后剩余数量保留在原释放轨道上，并以新的 `release_id` 记账，但 `source_event_id` 不变；剩余为零时该批次消失，且不会再释放。
- `sequester`：把等待物移入隔离账本并保留批次身份（`sequester_id`、原 `release_id`、`source_event_id`）。隔离期间既不释放也不进入倾向。
- `release_sequestered`：在指定时刻（或对指定 `sequester_id`）把仍存在的隔离物移回等待轨道并立即参与释放；新批次使用新 `release_id`，事件上以 `origin_release_id` 标明被隔离批次，源事件身份不变。
- `block_releases`/`allow_releases`：阻止或恢复释放，支持全局开关或针对单个 `release_id`。阻止不是销毁：被阻止的批次仍保留在等待账本中，恢复时刻仅仍存在且符合条件的数量释放（事件时刻为恢复时刻）；已经进入自由体系的部分无法被后来的阻止、失活或隔离追回，对已释放批次安排去向会直接报错。超出去往数量的申请同样报错，任何账本都不会出现负数。

同一时刻多类事件的处理顺序固定为：开放的释放（按 `release_id` 稳定排序）→ 竞争去向（按声明顺序）→ 条件变更。因此与释放同刻声明的阻断不能追回该次释放，同刻的失活/隔离仍按切换前的条件版本结算；`fate_events` 与每个命运事件都记录结算时的 `condition_version`、`fate_id` 和逐物种数量。从同一检查点 `fork()` 出的分支各自保留当时的条件与去向依据，逐事件比较互不影响。

保存点放在竞争事件前后同样与连续运行一致：命运与释放、条件变更一样作为可重放输入写入 `fate_schedule`，生效后写入 `fate_events`；预先抽取但尚未发生的反应若被竞争事件插入，其试探抽取连同随机数一并回滚重抽。旧版本状态 JSON（没有命运相关字段）可直接读取并续跑，旧的纯延迟释放轨迹与事件身份保持不变。

CLI 示例：

```bash
python -m biochem.cli run --model artifacts/model.json --state artifacts/t1.json --end-time 6 \
  --fate 2.5:inactivate:release=release-000001:amount=1:species=P \
  --fate 3.0:sequester:species=P \
  --fate 2.0:block_releases \
  --fate 4.0:allow_releases \
  --fate 5.0:release_sequestered
```

## 异常停止

运行中发生异常时，状态标记为 `status="failed"`，`failure_reason` 记录失败原因，数量与事件序列保持在最后一次可信状态；CLI 仍会保存状态文件并以非零码退出。
