"""ArviZ type definitions."""

from collections.abc import Hashable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

CoordSpec = Mapping[Hashable, Any]
DimSpec = Mapping[Hashable, Iterable[Hashable]]

DictData = Mapping[Hashable, "ArrayLike"]
