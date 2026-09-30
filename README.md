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
python -m biochem.cli run --model artifacts/model.json --state artifacts/t1.json --end-time 8 --rate-change 4 '{"bind":0.3}'
python -m biochem.cli experiments --model artifacts/model.json --runs 50 --duration 5
python -m biochem.cli deterministic --model artifacts/model.json --end-time 5 --step-size 0.05
```

第一次 `run` 使用种子生成轨迹；每个可信边界都会立刻抽取并保存下一次反应的发生时刻、反应编号、等待时间、被选倾向和总倾向。因此即使保存点落在下一次反应到来之前，恢复后也不会重新抽取已经决定的等待或反应。第二次直接读取同一个状态继续。

`--rate-change TIME RATE_JSON` 会把条件变更作为可重放输入记录在状态中。条件在指定时刻生效；若同一时刻已有反应，条件先生效，旧条件下抽中的未发生反应作废，并在新倾向下抽取后续等待。状态还会记录条件变更是否已应用。运行失败时会保留最后可信时间、数量、事件、随机数状态以及 `failed_at` 和 `failure_reason`。

每个反应事件都记录反应编号、发生时间、等待时间、被选倾向、总倾向、数量变化、选择原因和变化后数量。
