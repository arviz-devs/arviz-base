# pylint: disable=unused-argument
"""Utilities to generate labels from xarray objects."""

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, Protocol

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
    """Placeholder for type checking ``labeller`` used in docstrings."""

    def make_label_vert(  # noqa: D102
        self,
        var_name: str | None,
        sel: Mapping[Any, Hashable],
        isel: Mapping[Any, int | Sequence[int]],
    ) -> str: ...
    def make_label_flat(  # noqa: D102
        self,
        var_name: str | None,
        sel: Mapping[Any, Hashable],
        isel: Mapping[Any, int | Sequence[int]],
    ) -> str: ...


def mix_labellers(labellers, class_name="MixtureLabeller"):
    """Combine Labeller classes dynamically.

    The Labeller class aims to split plot labeling in ArviZ into atomic tasks to maximize
    extensibility, and the few classes provided are designed with small deviations
    from the base class, in many cases only one method is modified by the child class.
    It is to be expected then to want to use multiple classes "at once".

    This functions helps combine classes dynamically.

    For a general overview of ArviZ label customization, including
    ``mix_labellers``, see the :ref:`label_guide` page.

    Parameters
    ----------
    labellers : iterable of type
        Iterable of Labeller types to combine
    class_name : str, optional
        The name of the generated class

    Returns
    -------
    type
        Mixture class object. **It is not initialized**, and it should be
        initialized before passing it to ArviZ functions.

    Examples
    --------
    Combine the :class:`~arviz.labels.DimCoordLabeller` with the
    :class:`~arviz.labels.MapLabeller` to generate labels in the style of the
    ``DimCoordLabeller`` but using the mappings defined by ``MapLabeller``.
    Note that this works even though both modify the same methods because
    ``MapLabeller`` implements the mapping and then calls `super().method`.

    .. jupyter-execute::

        from arviz_base.labels import mix_labellers, DimCoordLabeller, MapLabeller
        l1 = DimCoordLabeller()
        sel = {"dim1": "a", "dim2": "top"}
        print(f"Output of DimCoordLabeller alone > {l1.sel_to_str(sel, sel)}")
        l2 = MapLabeller(dim_map={"dim1": "$d_1$", "dim2": r"$d_2$"})
        print(f"Output of MapLabeller alone > {l2.sel_to_str(sel, sel)}")
        l3 = mix_labellers(
            (MapLabeller, DimCoordLabeller)
        )(dim_map={"dim1": "$d_1$", "dim2": r"$d_2$"})
        print(f"Output of mixture labeller > {l3.sel_to_str(sel, sel)}")

    We can see how the mappings are taken into account as well as the dim+coord style. However,
    he order in the ``labellers`` arg iterator is important! See for yourself:

    .. jupyter-execute::

        l4 = mix_labellers(
            (DimCoordLabeller, MapLabeller)
        )(dim_map={"dim1": "$d_1$", "dim2": r"$d_2$"})
        print(f"Output of inverted mixture labeller > {l4.sel_to_str(sel, sel)}")

    """
    return type(class_name, labellers, {})


class BaseLabeller:
    """Base labeller class.

    The default labels for "theta" variable on the subset corresponding to "chain" 0
    and "school" "Name" are:

    .. code-block:: none
        :caption: Single line label

        theta[0, Name]

    .. code-block:: none
        :caption: Multi-line label

        theta
        0, Name

    See Also
    --------
    :ref:`label_guide`: Tutorial page on using labellers with ArviZ
    """

    def dim_coord_to_str(self, dim, coord_val, coord_idx):  # pylint: disable=no-self-use
        """Format a single dimension name, its value and positional indexes as a string.

        Parameters
        ----------
        dim : hashable
            Dimension name
        coord_val : hashable
            Coordinate label
        coord_idx : int or sequence of int
            Positions along `dim` where its coordinate is `coord_val`

        Returns
        -------
        str
        """
        return f"{coord_val}"

    def sel_to_str(self, sel, isel):
        """Format selection dictionaries as a string.

        Parameters
        ----------
        sel : mapping of {hashable_key : hashable}
        isel : mapping of {hashable_key : int or sequence of int}

        Returns
        -------
        str
        """
        if sel:
            return ", ".join(
                [
                    self.dim_coord_to_str(dim, v, i)
                    for (dim, v), (_, i) in zip(sel.items(), isel.items())
                ]
            )
        return ""

    def var_name_to_str(self, var_name):  # pylint: disable=no-self-use
        """Format a variable name as a string.

        Parameters
        ----------
        var_name : str or None
            The variable name. It should accept ``None``

        Returns
        -------
        str or None
        """
        return var_name

    def var_pp_to_str(self, var_name, pp_var_name):
        """Format the corresponding variable names for observation and posterior predictive.

        Parameters
        ----------
        var_name : str or None
        pp_var_name : str or None

        Returns
        -------
        str or None
        """
        var_name_str = self.var_name_to_str(var_name)
        pp_var_name_str = self.var_name_to_str(pp_var_name)
        if var_name_str is None and pp_var_name_str is None:
            return None
        if var_name_str == pp_var_name_str:
            return var_name_str
        return f"{var_name_str} / {pp_var_name_str}"

    def make_label_vert(self, var_name, sel, isel):
        """Format variable name and corresponding subset as a multiline string.

        Parameters
        ----------
        var_name : str or None
        sel : mapping of {hashable_key : hashable}
        isel : mapping of {hashable_key : int or sequence of int}

        Returns
        -------
        str
        """
        var_name_str = self.var_name_to_str(var_name)
        sel_str = self.sel_to_str(sel, isel)
        if not sel_str:
            return "" if var_name_str is None else var_name_str
        if var_name_str is None:
            return sel_str
        return f"{var_name_str}\n{sel_str}"

    def make_label_flat(self, var_name, sel, isel):
        """Format variable name and corresponding subset as a single line string.

        Parameters
        ----------
        var_name : str or None
        sel : mapping of {hashable_key : hashable}
        isel : mapping of {hashable_key : int or sequence of int}

        Returns
        -------
        str
        """
        var_name_str = self.var_name_to_str(var_name)
        sel_str = self.sel_to_str(sel, isel)
        if not sel_str:
            return "" if var_name_str is None else var_name_str
        if var_name_str is None:
            return sel_str
        return f"{var_name_str}[{sel_str}]"

    def make_pp_label(self, var_name, pp_var_name, sel, isel):
        """Format obs+pp variable name plus corresponding subsets as a multiline string.

        Parameters
        ----------
        var_name, pp_var_name : str or None
        sel : mapping of {hashable_key : hashable}
        isel : mapping of {hashable_key : int or sequence of int}

        Returns
        -------
        str
        """
        names = self.var_pp_to_str(var_name, pp_var_name)
        return self.make_label_vert(names, sel, isel)


class DimCoordLabeller(BaseLabeller):
    """Labeller class to show both dimension and coordinate value information.

    The default labels for "theta" variable on the subset corresponding to "chain" 0
    and "school" "Name" are:

    .. code-block:: none
        :caption: Single line label

        theta[chain: 0, school: Name]

    .. code-block:: none
        :caption: Multi-line label

        theta
        chain: 0, school: Name

    See Also
    --------
    :ref:`label_guide`: Tutorial page on using labellers with ArviZ
    """

    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        """Format a single dimension name, its value and positional indexes as a string.

        Parameters
        ----------
        dim : hashable
            Dimension name
        coord_val : hashable
            Coordinate label
        coord_idx : int or sequence of int
            Positions along `dim` where its coordinate is `coord_val`

        Returns
        -------
        str
        """
        return f"{dim}: {coord_val}"


class IdxLabeller(BaseLabeller):
    """Labeller class to show positional index information.

    The default labels for "theta" variable on the subset corresponding to "chain" 0
    and "school" "Name" (3rd and 5th element in the school coordinate values) are:

    .. code-block:: none
        :caption: Single line label

        theta[0, 2,4]

    .. code-block:: none
        :caption: Multi-line label

        theta
        0, 2,4

    See Also
    --------
    :ref:`label_guide`: Tutorial page on using labellers with ArviZ
    """

    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        """Format a single dimension name, its value and positional indexes as a string.

        Parameters
        ----------
        dim : hashable
            Dimension name
        coord_val : hashable
            Coordinate label
        coord_idx : int or sequence of int
            Positions along `dim` where its coordinate is `coord_val`

        Returns
        -------
        str
        """
        if not isinstance(coord_idx, Iterable):
            return f"{coord_idx}"
        return f"{','.join(str(idx) for idx in coord_idx)}"


class DimIdxLabeller(BaseLabeller):
    """Labeller class to show both dimension and positional index information.

    The default labels for "theta" variable on the subset corresponding to "chain" 0
    and "school" "Name" (3rd and 5th element in the school coordinate values) are:

    .. code-block:: none
        :caption: Single line label

        theta[chain#0, school#2,4]

    .. code-block:: none
        :caption: Multi-line label

        theta
        chain#0, school#2,4

    See Also
    --------
    :ref:`label_guide`: Tutorial page on using labellers with ArviZ
    """

    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        """Format a single dimension name, its value and positional indexes as a string.

        Parameters
        ----------
        dim : hashable
            Dimension name
        coord_val : hashable
            Coordinate label
        coord_idx : int or sequence of int
            Positions along `dim` where its coordinate is `coord_val`

        Returns
        -------
        str
        """
        if not isinstance(coord_idx, Iterable):
            return f"{dim}#{coord_idx}"
        return f"{dim}#{','.join(str(idx) for idx in coord_idx)}"


class MapLabeller(BaseLabeller):
    """Labeller class to perform provided replacements to elements when converting to string.

    It is a subclass of :class:`BaseLabeller` so the base behaviour is the same,
    but we can define replacements through dictionaries when initializing the class.

    The default labels for "theta" variable on the subset corresponding to "chain" 0
    and "school" "Name" with a replacement mapping on the variable name ``{"theta": "θ"}``
    and one on the coordinate values ``{"Name": "𝑁𝑎𝑚𝑒"}``

    .. code-block:: none
        :caption: Single line label

        θ[0, 𝑁𝑎𝑚𝑒]

    .. code-block:: none
        :caption: Multi-line label

        θ
        0, 𝑁𝑎𝑚𝑒

    See Also
    --------
    :ref:`label_guide`: Tutorial page on using labellers with ArviZ
    """

    def __init__(self, var_name_map=None, dim_map=None, coord_map=None):
        """Initialize a MapLabeller class.

        Parameters
        ----------
        var_name_map, dim_map : mapping of {hashable : hashable}, optional
            Keys are existing names and values are their respective desired labels.
        coord_map : mapping of {hashable : mapping of {hashable : hashable}}, optional
            The keys of the first level dictionary are dimension names, the inner
            dictionary has existing coord names as keys and their corresponding desired
            label as values.
        """
        self.var_name_map = {} if var_name_map is None else var_name_map
        self.dim_map = {} if dim_map is None else dim_map
        self.coord_map = {} if coord_map is None else coord_map

    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        """Format a single dimension name, its value and positional indexes as a string.

        Parameters
        ----------
        dim : hashable
            Dimension name
        coord_val : hashable
            Coordinate label
        coord_idx : int or sequence of int
            Positions along `dim` where its coordinate is `coord_val`

        Returns
        -------
        str
        """
        dim_str = self.dim_map.get(dim, dim)
        coord_str = self.coord_map.get(dim, {}).get(coord_val, coord_val)
        return super().dim_coord_to_str(dim_str, coord_str, coord_idx)

    def var_name_to_str(self, var_name):
        """Format a variable name as a string.

        Parameters
        ----------
        var_name : str or None
            The variable name. It should accept ``None``

        Returns
        -------
        str or None
        """
        var_name_str = self.var_name_map.get(var_name, var_name)
        return super().var_name_to_str(var_name_str)


class NoVarLabeller(BaseLabeller):
    """Labeller class to exclude the variable name from the generated labels.

    The default labels for "theta" variable on the subset corresponding to "chain" 0
    and "school" "Name" are:

    .. code-block:: none
        :caption: Single line label

        0, Name

    .. code-block:: none
        :caption: Multi-line label

        0, Name

    See Also
    --------
    :ref:`label_guide`: Tutorial page on using labellers with ArviZ
    """

    def var_name_to_str(self, var_name):
        """Format a variable name as a string.

        Parameters
        ----------
        var_name : str or None
            The variable name. It should accept ``None``

        Returns
        -------
        str or None
        """
        return None
