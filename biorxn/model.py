"""物种、反应和网络模型。

本模块只描述“有什么、怎么连”，不包含任何模拟算法；
随机运行器和后续确定性求解器都读取这里的同一份模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Species:
    id: str
    name: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name or self.id}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Species":
        return cls(id=str(data["id"]), name=str(data.get("name", data["id"])))


@dataclass(frozen=True)
class Reaction:
    """一条基元反应。

    reactants/products 为 物种ID -> 正整数化学计量数；
    rate 是随机学速率常数，由已注册规则解释。
    """

    id: str
    reactants: dict[str, int] = field(default_factory=dict)
    products: dict[str, int] = field(default_factory=dict)
    rate: float = 0.0
    name: str = ""
    rule: str = "mass_action"

    def changes(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for species_id, amount in self.products.items():
            result[species_id] = result.get(species_id, 0) + amount
        for species_id, amount in self.reactants.items():
            result[species_id] = result.get(species_id, 0) - amount
        return {key: value for key, value in result.items() if value != 0}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name or self.id,
            "rule": self.rule,
            "rate": self.rate,
            "reactants": dict(self.reactants),
            "products": dict(self.products),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Reaction":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            rule=str(data.get("rule", "mass_action")),
            rate=float(data["rate"]),
            reactants={k: int(v) for k, v in dict(data["reactants"]).items()},
            products={k: int(v) for k, v in dict(data["products"]).items()},
        )


class ValidationError(ValueError):
    pass


@dataclass
class Model:
    id: str = "model"
    version: int = 1
    metadata: dict = field(default_factory=dict)

    def __init__(
        self,
        species: Iterable[Species | str],
        reactions: Iterable[Reaction],
        id: str = "model",
        metadata: Mapping | None = None,
    ) -> None:
        self.id = id
        self.species = {}
        for item in species:
            sp = Species(item) if isinstance(item, str) else item
            if sp.id in self.species:
                raise ValidationError(f"物种ID重复: {sp.id}")
            self.species[sp.id] = sp
        self.reactions = {}
        for reaction in reactions:
            if reaction.id in self.reactions:
                raise ValidationError(f"反应ID重复: {reaction.id}")
            self.reactions[reaction.id] = reaction
        self.version = 1
        self.metadata = dict(metadata or {})
        self.validate()

    @property
    def species_ids(self) -> list[str]:
        return list(self.species)

    def validate(self) -> None:
        for reaction in self.reactions.values():
            if reaction.rate < 0:
                raise ValidationError(f"反应 {reaction.id} 速率不能为负")
            for side_name, side in (
                ("反应物", reaction.reactants),
                ("产物", reaction.products),
            ):
                for species_id, amount in side.items():
                    if species_id not in self.species:
                        raise ValidationError(
                            f"反应 {reaction.id} 的{side_name}引用未知物种: {species_id}"
                        )
                    if not isinstance(amount, int) or amount <= 0:
                        raise ValidationError(
                            f"反应 {reaction.id} 的化学计量数必须是正整数: {species_id}={amount}"
                        )
            if not reaction.changes():
                raise ValidationError(f"反应 {reaction.id} 不产生任何数量变化")

    def stoichiometric_matrix(self) -> list[list[int]]:
        """返回 species x reaction 的整数净化学计量矩阵。"""
        return [
            [reaction.changes().get(species_id, 0) for reaction in self.reactions.values()]
            for species_id in self.species_ids
        ]

    def conservation_laws(self) -> list[dict[str, int]]:
        """求线性守恒量 S.T @ c = 0 的整数基（如原子/总酶守恒）。"""
        n_species = len(self.species)
        n_reactions = len(self.reactions)
        matrix = self.stoichiometric_matrix()
        rows = [
            [Fraction(matrix[row][col]) for col in range(n_reactions)]
            + [Fraction(1 if i == row else 0) for i in range(n_species)]
            for row in range(n_species)
        ]
        pivot_row = 0
        for col in range(n_reactions):
            candidate = next(
                (r for r in range(pivot_row, n_species) if rows[r][col] != 0), None
            )
            if candidate is None:
                continue
            rows[pivot_row], rows[candidate] = rows[candidate], rows[pivot_row]
            pivot = rows[pivot_row][col]
            rows[pivot_row] = [value / pivot for value in rows[pivot_row]]
            for r in range(n_species):
                if r != pivot_row and rows[r][col] != 0:
                    factor = rows[r][col]
                    rows[r] = [
                        value - factor * pivot_value
                        for value, pivot_value in zip(rows[r], rows[pivot_row])
                    ]
            pivot_row += 1

        laws = []
        for row in range(pivot_row, n_species):
            coefficients = rows[row][n_reactions:]
            scale = 1
            for value in coefficients:
                scale = scale * value.denominator // _gcd(scale, value.denominator)
            integers = [int(value * scale) for value in coefficients]
            divisor = 0
            for value in integers:
                divisor = _gcd(divisor, abs(value))
            if divisor > 1:
                integers = [value // divisor for value in integers]
            laws.append(dict(zip(self.species_ids, integers)))
        return laws

    def to_dict(self) -> dict:
        return {
            "schema": "biorxn.model/v1",
            "id": self.id,
            "version": self.version,
            "metadata": self.metadata,
            "species": [sp.to_dict() for sp in self.species.values()],
            "reactions": [rxn.to_dict() for rxn in self.reactions.values()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Model":
        model = cls(
            species=[Species.from_dict(item) for item in data["species"]],
            reactions=[Reaction.from_dict(item) for item in data["reactions"]],
            id=str(data.get("id", "model")),
            metadata=dict(data.get("metadata", {})),
        )
        model.version = int(data.get("version", 1))
        return model


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a
