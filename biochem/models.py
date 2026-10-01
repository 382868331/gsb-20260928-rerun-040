"""Serializable experiment model definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Species:
    id: str
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name or self.id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Species":
        return cls(id=str(data["id"]), name=str(data.get("name", data["id"])))


@dataclass(frozen=True)
class Reaction:
    id: str
    reactants: dict[str, int] = field(default_factory=dict)
    products: dict[str, int] = field(default_factory=dict)
    rate_constant: float = 1.0
    name: str = ""
    # species_id -> (amount produced, delay before it enters the free pool)
    delayed_products: dict[str, tuple[int, float]] = field(default_factory=dict)

    @property
    def stoichiometry(self) -> dict[str, int]:
        species = set(self.reactants) | set(self.products) | set(self.delayed_products)
        return {
            species_id: self.products.get(species_id, 0)
            - self.reactants.get(species_id, 0)
            for species_id in species
        }

    @property
    def initiation_stoichiometry(self) -> dict[str, int]:
        """Free-count change applied immediately when the reaction starts.

        Delayed products are withheld in the waiting pool and therefore do
        not appear here until their scheduled release.
        """

        species = set(self.reactants) | set(self.products)
        return {
            species_id: self.products.get(species_id, 0)
            - self.reactants.get(species_id, 0)
            for species_id in species
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name or self.id,
            "reactants": dict(self.reactants),
            "products": dict(self.products),
            "rate_constant": self.rate_constant,
            "delayed_products": {
                species_id: {"amount": amount, "delay": delay}
                for species_id, (amount, delay) in self.delayed_products.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Reaction":
        delayed_raw = data.get("delayed_products", {})
        delayed_products = {
            str(species_id): (int(item["amount"]), float(item["delay"]))
            for species_id, item in delayed_raw.items()
        }
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            reactants={str(k): int(v) for k, v in data.get("reactants", {}).items()},
            products={str(k): int(v) for k, v in data.get("products", {}).items()},
            rate_constant=float(data.get("rate_constant", 1.0)),
            delayed_products=delayed_products,
        )


@dataclass(frozen=True)
class ExperimentModel:
    species: tuple[Species, ...]
    reactions: tuple[Reaction, ...]
    initial_counts: dict[str, int]
    name: str = "biochemical-experiment"

    @property
    def species_ids(self) -> set[str]:
        return {species.id for species in self.species}

    def reaction(self, reaction_id: str) -> Reaction:
        for reaction in self.reactions:
            if reaction.id == reaction_id:
                return reaction
        raise KeyError(f"unknown reaction: {reaction_id}")

    def validate(self) -> None:
        ids = self.species_ids
        if len(ids) != len(self.species):
            raise ValueError("species ids must be unique")
        if not self.species:
            raise ValueError("model must contain at least one species")
        reaction_ids = [reaction.id for reaction in self.reactions]
        if len(reaction_ids) != len(set(reaction_ids)):
            raise ValueError("reaction ids must be unique")
        for reaction in self.reactions:
            if reaction.rate_constant < 0:
                raise ValueError(f"reaction {reaction.id} has a negative rate constant")
            for side_name, side in (("reactants", reaction.reactants), ("products", reaction.products)):
                for species_id, amount in side.items():
                    if species_id not in ids:
                        raise ValueError(f"{side_name} references unknown species {species_id}")
                    if amount <= 0:
                        raise ValueError(f"stoichiometry must be positive for {reaction.id}/{species_id}")
            for species_id, (amount, delay) in reaction.delayed_products.items():
                if species_id not in ids:
                    raise ValueError(f"delayed product references unknown species {species_id}")
                if amount <= 0:
                    raise ValueError(f"delayed amount must be positive for {reaction.id}/{species_id}")
                if delay <= 0:
                    raise ValueError(f"delayed delay must be positive for {reaction.id}/{species_id}")
        for species_id, count in self.initial_counts.items():
            if species_id not in ids:
                raise ValueError(f"initial count references unknown species {species_id}")
            if count < 0:
                raise ValueError(f"initial count must be non-negative for {species_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "species": [species.to_dict() for species in self.species],
            "reactions": [reaction.to_dict() for reaction in self.reactions],
            "initial_counts": dict(self.initial_counts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentModel":
        version = int(data.get("schema_version", 1))
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version: {version}")
        model = cls(
            name=str(data.get("name", "biochemical-experiment")),
            species=tuple(Species.from_dict(item) for item in data.get("species", [])),
            reactions=tuple(Reaction.from_dict(item) for item in data.get("reactions", [])),
            initial_counts={
                str(k): int(v) for k, v in data.get("initial_counts", {}).items()
            },
        )
        model.validate()
        return model


def build_conservation_network() -> ExperimentModel:
    """Create a finite enzyme network with E and substrate-atom conservation."""
    species = tuple(
        Species(species_id, name)
        for species_id, name in (
            ("E", "free enzyme"),
            ("S", "substrate"),
            ("ES", "enzyme-substrate complex"),
            ("P", "product"),
        )
    )
    reactions = (
        Reaction("bind", {"E": 1, "S": 1}, {"ES": 1}, 0.12, "substrate binding"),
        Reaction("unbind", {"ES": 1}, {"E": 1, "S": 1}, 0.04, "complex dissociation"),
        Reaction("catalyze", {"ES": 1}, {"E": 1, "P": 1}, 0.20, "product formation"),
    )
    return ExperimentModel(
        species=species,
        reactions=reactions,
        initial_counts={"E": 3, "S": 12, "ES": 0, "P": 0},
        name="conserved-enzyme-network",
    )


def build_delayed_release_network() -> ExperimentModel:
    """Enzyme network whose product enters the free pool after a delay.

    ``catalyze`` immediately frees the enzyme and puts one product molecule
    into the waiting pool; the product is only released 0.8 time units later
    and cannot participate in anything before its release.
    """

    species = tuple(
        Species(species_id, name)
        for species_id, name in (
            ("E", "free enzyme"),
            ("S", "substrate"),
            ("ES", "enzyme-substrate complex"),
            ("P", "product"),
        )
    )
    reactions = (
        Reaction("bind", {"E": 1, "S": 1}, {"ES": 1}, 0.12, "substrate binding"),
        Reaction("unbind", {"ES": 1}, {"E": 1, "S": 1}, 0.04, "complex dissociation"),
        Reaction(
            "catalyze",
            {"ES": 1},
            {"E": 1},
            0.20,
            "product formation with delayed release",
            delayed_products={"P": (1, 0.8)},
        ),
    )
    return ExperimentModel(
        species=species,
        reactions=reactions,
        initial_counts={"E": 3, "S": 12, "ES": 0, "P": 0},
        name="delayed-release-enzyme-network",
    )
