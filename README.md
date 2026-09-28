# 生化反应实验系统（第一版）

纯 Python 标准库实现的小规模生化反应实验框架。系统只做离线模拟，不连接实验设备，也不绘制或回放预设曲线。

## 结构

- biorxn/model.py：物种、反应、化学计量矩阵和整数守恒关系。
- biorxn/rules.py：反应规则注册表；默认规则为离散质量作用倾向度。
- biorxn/runner.py：Gillespie SSA 随机运行器。
- biorxn/records.py：每次变化一条事件，记录时间、原因、变化前后数量和随机数。
- biorxn/persistence.py：模型、记录、种子和 RNG 中途状态的 JSON 保存/恢复。
- biorxn/deterministic.py：读取同一模型的显式欧拉确定性计算。
- biorxn/experiments.py：多次独立短实验、平均变化和 CSV 输出。
- biorxn/networks.py：人工守恒网络，包含酶结合、解离、催化和回收反应。

## 关键约束

- 任一反应物数量不足时，对应反应倾向度为 0，反应不能发生。
- 每次反应都保存 time、reaction_id、reason、before、change、after。
- 快照保存 Python random.Random 内部状态；恢复后继续同一条随机轨迹。
- 随机运行器和确定性求解器都读取同一个 Model。
- 人工网络自动求出守恒基；演示中核对总酶量 E + ES 和底物池相关守恒量。

## 运行

    .venv/bin/python -m pytest -q
    .venv/bin/python scripts/run_demo.py

演示会在 outputs/ 生成模型、快照、逐次变化记录、200 次短实验汇总和每次重复实验的 CSV。

短实验从相同初始状态、不同种子独立启动，统计平均净变化和单位时间变化率；确定性欧拉结果只作为同模型的独立数学对照，不参与生成随机轨迹。
