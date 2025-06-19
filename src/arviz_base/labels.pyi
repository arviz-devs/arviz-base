# File generated with docstub

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Protocol

__all__ = [
    "mix_labellers",
    "Labeller",
    "BaseLabeller",
    "DimCoordLabeller",
    "IdxLabeller",
    "DimIdxLabeller",
    "MapLabeller",
    "NoVarLabeller",
]

class Labeller(Protocol):
    def make_label_vert(
        self,
        var_name: str | None,
        sel: Mapping[Hashable, Hashable],
        isel: Mapping[Hashable, int | Sequence[int]],
    ) -> str: ...
    def make_label_flat(
        self,
        var_name: str | None,
        sel: Mapping[Hashable, Hashable],
        isel: Mapping[Hashable, int | Sequence[int]],
    ) -> str: ...

def mix_labellers(labellers: Iterable[type], class_name: str = ...) -> type: ...

class BaseLabeller:
    def dim_coord_to_str(
        self, dim: Hashable, coord_val: Hashable, coord_idx: int | Sequence[int]
    ) -> str: ...
    def sel_to_str(
        self,
        sel: Mapping[Hashable, Hashable],
        isel: Mapping[Hashable, int | Sequence[int]],
    ) -> str: ...
    def var_name_to_str(self, var_name: str | None) -> str | None: ...
    def var_pp_to_str(
        self, var_name: str | None, pp_var_name: str | None
    ) -> str | None: ...
    def make_label_vert(
        self,
        var_name: str | None,
        sel: Mapping[Hashable, Hashable],
        isel: Mapping[Hashable, int | Sequence[int]],
    ) -> str: ...
    def make_label_flat(
        self,
        var_name: str | None,
        sel: Mapping[Hashable, Hashable],
        isel: Mapping[Hashable, int | Sequence[int]],
    ) -> str: ...
    def make_pp_label(
        self,
        var_name,
        pp_var_name,
        sel: Mapping[Hashable, Hashable],
        isel: Mapping[Hashable, int | Sequence[int]],
    ) -> str: ...

class DimCoordLabeller(BaseLabeller):
    def dim_coord_to_str(
        self, dim: Hashable, coord_val: Hashable, coord_idx: int | Sequence[int]
    ) -> str: ...

class IdxLabeller(BaseLabeller):
    def dim_coord_to_str(
        self, dim: Hashable, coord_val: Hashable, coord_idx: int | Sequence[int]
    ) -> str: ...

class DimIdxLabeller(BaseLabeller):
    def dim_coord_to_str(
        self, dim: Hashable, coord_val: Hashable, coord_idx: int | Sequence[int]
    ) -> str: ...

class MapLabeller(BaseLabeller):
    def __init__(
        self,
        var_name_map=...,
        dim_map=...,
        coord_map: Mapping[Hashable, Mapping[Hashable, Hashable]] | None = ...,
    ) -> None: ...
    def dim_coord_to_str(
        self, dim: Hashable, coord_val: Hashable, coord_idx: int | Sequence[int]
    ) -> str: ...
    def var_name_to_str(self, var_name: str | None) -> str | None: ...

class NoVarLabeller(BaseLabeller):
    def var_name_to_str(self, var_name: str | None) -> str | None: ...
