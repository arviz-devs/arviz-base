# File generated with docstub

import os
import re
from collections.abc import Callable

from _typeshed import Incomplete

def citations(
    methods: Callable | list[Callable] | None = ...,
    filepath: str | None = ...,
    format_type: str = ...,
) -> None: ...
def _extract_ids_per_entry(data: Incomplete, text: Incomplete) -> None: ...
def _find_bibtex_entries(header: Incomplete, data: Incomplete) -> None: ...
def _get_header(methods: Incomplete = ...) -> None: ...
