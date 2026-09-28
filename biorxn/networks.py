"""内置人工反应网络。

网络是构造出来的，不连接任何实验设备，也不包含预设结果曲线。
"""

from __future__ import annotations

from .model import Model, Reaction, Species


def enzyme_toggle_network() -> Model:
    """小型酶促人工网络：

    E + S <-> ES -> E + P
    P -> S

    守恒量包括 S+ES+P（底物原子池）和 E+ES（总酶量）。
    """
    species = [
        Species("E", "游离酶"),
        Species("S", "底物"),
        Species("ES", "酶底物复合物"),
        Species("P", "产物"),
    ]
    reactions = [
        Reaction(
            id="bind",
            name="结合",
            reactants={"E": 1, "S": 1},
            products={"ES": 1},
            rate=0.08,
        ),
        Reaction(
            id="unbind",
            name="解离",
            reactants={"ES": 1},
            products={"E": 1, "S": 1},
            rate=0.15,
        ),
        Reaction(
            id="catalyze",
            name="催化",
            reactants={"ES": 1},
            products={"E": 1, "P": 1},
            rate=0.35,
        ),
        Reaction(
            id="recycle",
            name="产物回收",
            reactants={"P": 1},
            products={"S": 1},
            rate=0.05,
        ),
    ]
    return Model(
        species=species,
        reactions=reactions,
        id="enzyme_toggle",
        metadata={"description": "可逆酶促、催化与回收的人工守恒网络"},
    )


def initial_enzyme_toggle() -> dict[str, int]:
    return {"E": 10, "S": 60, "ES": 0, "P": 0}
