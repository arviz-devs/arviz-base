"""ArviZ type definitions."""

from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

CoordSpec = Mapping[Any, "ArrayLike"]
CoordOut = dict[Any, "NDArray"]
DimSpec = Mapping[Any, Sequence[Hashable]]
DimOut = dict[Any, list[Hashable]]

DictData = Mapping[Any, "ArrayLike"]
