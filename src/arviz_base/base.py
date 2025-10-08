"""ArviZ basic functions and converters."""

import datetime
import importlib
import re
import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import xarray as xr

from arviz_base._version import __version__
from arviz_base.rcparams import rcParams

if TYPE_CHECKING:
    pass

RequiresArgTypeT = TypeVar("RequiresArgTypeT")
RequiresReturnTypeT = TypeVar("RequiresReturnTypeT")


def generate_dims_coords(
    shape,
    var_name,
    dims=None,
    coords=None,
    index_origin=None,
    skip_event_dims=False,
    check_conventions=True,
):
    """Generate default dimensions and coordinates for a variable.

    Parameters
    ----------
    shape : iterable of int
        Shape of the variable
    var_name : hashable
        Name of the variable. If no dimension name(s) is provided, ArviZ
        will generate a default dimension name using ``var_name``, e.g.
        ``"foo_dim_0"`` for the first dimension if ``var_name`` is ``"foo"``.
    dims : iterable of hashable, optional
        Dimension names (or identifiers) for the variable.
        If `skip_event_dims` is ``True`` it can be longer than `shape`.
        In that case, only the first ``len(shape)`` elements in `dims` will be used.
        Moreover, if needed, axis of length 1 in shape will also be given
        different names than the ones provided in `dims`.
    coords : dict of {hashable_key : array_like}, optional
        Map of dimension names to coordinate values. Dimensions without coordinate
        values mapped to them will be given an integer range as coordinate values.
        It can have keys for dimension names not present in that variable.
    index_origin : int, optional
        Starting value of generated integer coordinate values.
        Defaults to the value in rcParam ``data.index_origin``.
    skip_event_dims : bool, default False
        Whether to allow for different sizes between `shape` and `dims`.
        See description in `dims` for more details.
    check_conventions : bool, optional
        Check ArviZ conventions. Per the ArviZ schema, some dimension names
        have specific meaning and there might be inconsistencies caught here
        in the dimension naming step.


    Returns
    -------
    dims : list of hashable
        Default dims for that variable
    coords : dict of {hashable_key : ndarray}
        Default coords for that variable
    """
    if index_origin is None:
        index_origin = rcParams["data.index_origin"]
    if dims is None:
        dims = []

    if coords is None:
        coords = {}

    coords = deepcopy(coords)
    dims = deepcopy(dims)

    if len(dims) > len(shape):
        if skip_event_dims:
            dims = dims[: len(shape)]
        else:
            raise ValueError(
                (
                    "In variable {var_name}, there are "
                    + "more dims ({dims_len}) given than existing ones ({shape_len}). "
                    + "dims and shape should match with `skip_event_dims=False`"
                ).format(
                    var_name=var_name,
                    dims_len=len(dims),
                    shape_len=len(shape),
                )
            )
    if skip_event_dims:
        # In some cases, even when there is an event dim, the shape has the
        # right length but the length of the axis doesn't match.
        # For example, the log likelihood of a 3d MvNormal with 20 observations
        # should be (20,) but it can also be (20, 1). The code below ensures
        # the (20, 1) option also works.
        for i, (dim, dim_size) in enumerate(zip(dims, shape)):
            if (dim in coords) and (dim_size != len(coords[dim])):
                dims = dims[:i]
                break

    missing_dim_count = 0
    for idx, dim_len in enumerate(shape):
        if idx + 1 > len(dims):
            dim_name = f"{var_name}_dim_{missing_dim_count}"
            missing_dim_count += 1
            dims.append(dim_name)
        elif dims[idx] is None:
            dim_name = f"{var_name}_dim_{missing_dim_count}"
            missing_dim_count += 1
            dims[idx] = dim_name
        dim_name = dims[idx]
        if dim_name not in coords:
            coords[dim_name] = np.arange(index_origin, dim_len + index_origin)
    coords = {dim_name: coords[dim_name] for dim_name in dims}
    if check_conventions:
        short_long_pairs = (("draw", "chain"), ("draw", "pred_id"), ("sample", "pred_id"))
        for long_dim, short_dim in short_long_pairs:
            if (
                long_dim in dims
                and short_dim in dims
                and len(coords[short_dim]) > len(coords[long_dim])
            ):
                warnings.warn(
                    f"Found {short_dim} dimension to be longer than {long_dim} dimension, "
                    "check dimensions are correctly named.",
                    UserWarning,
                )
        if "sample" in dims and (("draw" in dims) or ("chain" in dims)):
            warnings.warn(
                "Found dimension named 'sample' alongside 'chain'/'draw' ones, "
                "check dimensions are correctly named.",
                UserWarning,
            )
    return dims, coords


def ndarray_to_dataarray(
    ary,
    var_name,
    *,
    dims=None,
    sample_dims=None,
    coords=None,
    index_origin=None,
    skip_event_dims=False,
    check_conventions=True,
):
    """Convert a numpy array to an xarray.DataArray.

    The conversion considers some ArviZ conventions and adds extra
    attributes, so it is similar to initializing an :class:`xarray.DataArray`
    but not equivalent.

    Parameters
    ----------
    ary : scalar or array_like
        Values for the DataArray object to be created.
    var_name : hashable
        Name of the created DataArray object.
    dims : iterable of hashable, optional
        Dimensions of the DataArray.
    coords : dict of {hashable_key : array_like}, optional
        Coordinates for the dataarray
    sample_dims : iterable of hashable, optional
        Dimensions that should be assumed to be present.
        If missing, they will be added as the dimensions corresponding to the
        leading axes.
    index_origin : int, optional
        Passed to :func:`generate_dims_coords`
    skip_event_dims : bool, optional
        Passed to :func:`generate_dims_coords`
    check_conventions : bool, optional
        Check ArviZ conventions. Per the ArviZ schema, some dimension names
        have specific meaning and there might be inconsistencies caught here
        in the dimension naming step.

    Returns
    -------
    DataArray

    See Also
    --------
    dict_to_dataset
    """
    if dims is None:
        dims = []

    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]

    if sample_dims:
        var_dims = [sample_dim for sample_dim in sample_dims if sample_dim not in dims]
        var_dims.extend(dims)
    else:
        var_dims = dims
    var_dims, var_coords = generate_dims_coords(
        ary.shape if hasattr(ary, "shape") else (),
        var_name=var_name,
        dims=var_dims,
        coords=coords,
        index_origin=index_origin,
        skip_event_dims=skip_event_dims,
        check_conventions=check_conventions,
    )
    return xr.DataArray(ary, coords=var_coords, dims=var_dims)


def dict_to_dataset(
    data,
    *,
    attrs=None,
    inference_library=None,
    coords=None,
    dims=None,
    sample_dims=None,
    index_origin=None,
    skip_event_dims=False,
    check_conventions=True,
):
    """Convert a dictionary of numpy arrays to an xarray.Dataset.

    The conversion considers some ArviZ conventions and adds extra
    attributes, so it is similar to initializing an :class:`xarray.Dataset`
    but not equivalent.

    Parameters
    ----------
    data : mapping of {hashable_key : array_like}
        Data to convert. Keys are variable names.
    attrs : mapping of {hashable_key : any}, optional
        JSON-like arbitrary metadata to attach to the dataset, in addition to default
        attributes added by :func:`make_attrs`.

        .. note::

           No serialization checks are done in this function, so you might generate
           :class:`~xarray.Dataset` objects that can't be serialized or that can
           only be serialized to some backends.

    inference_library : module, optional
        Library used for performing inference. Will be included in the
        :class:`xarray.Dataset` attributes.
    coords : dict of {hashable_key : array_like}, optional
        Coordinates for the dataset
    dims : dict of {hashable : sequence of hashable}, optional
        Dimensions of each variable. The keys are variable names, values are lists of
        coordinates.
    sample_dims : sequence of hashable, optional
        Dimensions that should be assumed to be present in _all_ variables.
        If missing, they will be added as the dimensions corresponding to the
        leading axes.
    index_origin : int, optional
        Passed to :func:`generate_dims_coords`
    skip_event_dims : bool, optional
        Passed to :func:`generate_dims_coords`
    check_conventions : bool, optional
        Check ArviZ conventions. Per the ArviZ schema, some dimension names
        have specific meaning and there might be inconsistencies caught here
        in the dimension naming step.

    Returns
    -------
    Dataset

    See Also
    --------
    ndarray_to_dataarray
    convert_to_dataset
        General conversion to `xarray.Dataset` via :func:`convert_to_datatree`

    Examples
    --------
    Generate a :class:`~xarray.Dataset` with two variables
    using ``sample_dims``:

    .. jupyter-execute::

        import arviz_base as az
        import numpy as np
        rng = np.random.default_rng(2)
        az.dict_to_dataset(
            {"a": rng.normal(size=(4, 100)), "b": rng.normal(size=(4, 100))},
            sample_dims=["chain", "draw"],
        )

    Generate a :class:`~xarray.Dataset` with the ``chain`` and ``draw``
    dimensions in different position. Setting the dimensions for ``a``
    to "group" and "chain", ``sample_dims`` will then be used to prepend
    the "draw" dimension only as "chain" is already there.

    .. jupyter-execute::

        az.dict_to_dataset(
            {"a": rng.normal(size=(10, 5, 4)), "b": rng.normal(size=(10, 4))},
            dims={"a": ["group", "chain"]},
            sample_dims=["draw", "chain"],
        )

    """
    if dims is None:
        dims = {}

    data_vars = {
        var_name: ndarray_to_dataarray(
            values,
            var_name=var_name,
            dims=dims.get(var_name, []),
            sample_dims=sample_dims,
            coords=coords,
            index_origin=index_origin,
            skip_event_dims=skip_event_dims,
            check_conventions=check_conventions,
        )
        for var_name, values in data.items()
    }

    return xr.Dataset(
        data_vars=data_vars, attrs=make_attrs(attrs=attrs, inference_library=inference_library)
    )


def make_attrs(attrs=None, inference_library=None):
    """Make standard attributes to attach to xarray datasets.

    Parameters
    ----------
    attrs : mapping of {hashable_key : any}, optional
        Additional attributes to add or overwrite
    inference_library : module, optional
        Library used to perform inference.

    Returns
    -------
    dict
        attrs
    """
    default_attrs = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "creation_library": "ArviZ",
        "creation_library_version": __version__,
        "creation_library_language": "Python",
    }
    if inference_library is not None:
        library_name = inference_library.__name__
        default_attrs["inference_library"] = library_name
        try:
            version = importlib.metadata.version(library_name)
            default_attrs["inference_library_version"] = version
        except importlib.metadata.PackageNotFoundError:
            if hasattr(inference_library, "__version__"):
                version = inference_library.__version__
                default_attrs["inference_library_version"] = version
    if attrs is not None:
        default_attrs.update(attrs)
    return default_attrs


class requires:  # pylint: disable=invalid-name
    """Decorator to return None if an object does not have the required attribute.

    If the decorator is called various times on the same function with different
    attributes, it will return None if one of them is missing. If instead a list
    of attributes is passed, it will return None if all attributes in the list are
    missing. Both functionalities can be combined as desired.

    It can only be used to decorate functions/methods with a single argument,
    e.g. ``posterior_to_xarray(self)`` is valid,
    but ``posterior_to_xarray(self, other_arg)`` would not be.
    See https://github.com/arviz-devs/arviz/pull/1504 for more discussion.
    """

    def __init__(self, *props: str | list[str]) -> None:
        self.props: tuple[str | list[str], ...] = props

    def __call__(
        self, func: Callable[[RequiresArgTypeT], RequiresReturnTypeT]
    ) -> Callable[[RequiresArgTypeT], RequiresReturnTypeT | None]:  # noqa: D202
        """Wrap the decorated function."""

        def wrapped(cls: RequiresArgTypeT) -> RequiresReturnTypeT | None:
            """Return None if not all props are available."""
            for prop in self.props:
                prop_list = [prop] if isinstance(prop, str) else prop
                if all(getattr(cls, prop_i) is None for prop_i in prop_list):
                    return None
            return func(cls)

        return wrapped


def infer_stan_dtypes(stan_code):
    """Infer Stan integer variables from generated quantities block."""
    # Remove old deprecated comments
    stan_code = "\n".join(
        line if "#" not in line else line[: line.find("#")] for line in stan_code.splitlines()
    )
    pattern_remove_comments = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"', re.DOTALL | re.MULTILINE
    )
    stan_code = re.sub(pattern_remove_comments, "", stan_code)

    # Check generated quantities
    if "generated quantities" not in stan_code:
        return {}

    # Extract generated quantities block
    gen_quantities_location = stan_code.index("generated quantities")
    block_start = gen_quantities_location + stan_code[gen_quantities_location:].index("{")

    curly_bracket_count = 0
    block_end = None
    for block_end, char in enumerate(stan_code[block_start:], block_start + 1):
        if char == "{":
            curly_bracket_count += 1
        elif char == "}":
            curly_bracket_count -= 1

            if curly_bracket_count == 0:
                break

    stan_code = stan_code[block_start:block_end]

    stan_integer = r"int"
    stan_limits = r"(?:\<[^\>]+\>)*"  # ignore group: 0 or more <....>
    stan_param = r"([^;=\s\[]+)"  # capture group: ends= ";", "=", "[" or whitespace
    stan_ws = r"\s*"  # 0 or more whitespace
    stan_ws_one = r"\s+"  # 1 or more whitespace
    pattern_int = re.compile(
        "".join((stan_integer, stan_ws_one, stan_limits, stan_ws, stan_param)), re.IGNORECASE
    )
    dtypes = {key.strip(): "int" for key in re.findall(pattern_int, stan_code)}
    return dtypes
