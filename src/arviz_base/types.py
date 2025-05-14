"""ArviZ type definitions."""

from collections.abc import Hashable, Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

CoordSpec = Mapping[Hashable, "ArrayLike"]
CoordOut = dict[Hashable, "NDArray"]
DimSpec = Mapping[Hashable, Iterable[Hashable]]
DimOut = dict[Hashable, list[Hashable]]

DictData = Mapping[Hashable, "ArrayLike"]
