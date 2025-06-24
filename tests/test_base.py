# pylint: disable=no-member, no-self-use, invalid-name, redefined-outer-name

import os
from collections import namedtuple
from collections.abc import Hashable, Iterable
from numbers import Number
from urllib.parse import urlunsplit

import numpy as np
import pytest
from numpy.typing import ArrayLike
from xarray import DataTree
from xarray.testing import assert_allclose

from arviz_base import (
    dict_to_dataset,
    generate_dims_coords,
    list_datasets,
    load_arviz_data,
    make_attrs,
    ndarray_to_dataarray,
)
from arviz_base.datasets import LOCAL_DATASETS, REMOTE_DATASETS, RemoteFileMetadata, clear_data_home

netcdf_nightlies_skip = pytest.mark.skipif(
    os.environ.get("NIGHTLIES", "FALSE") == "TRUE",
    reason="Skip netcdf4 dependent tests from nightlies as it generally takes longer to update.",
)


@pytest.fixture(autouse=True)
def no_remote_data(monkeypatch, tmpdir):
    """Delete all remote data and replace it with a local dataset."""
    keys = list(REMOTE_DATASETS)
    for key in keys:
        monkeypatch.delitem(REMOTE_DATASETS, key)

    centered = LOCAL_DATASETS["centered_eight"]
    filename = os.path.join(str(tmpdir), os.path.basename(centered.filename))

    url = urlunsplit(("file", "", centered.filename, "", ""))

    monkeypatch.setitem(
        REMOTE_DATASETS,
        "test_remote",
        RemoteFileMetadata(
            name="test_remote",
            filename=filename,
            url=url,
            checksum="8efc3abafe0c796eb9aea7b69490d4e2400a33c57504ef4932e1c7105849176f",
            description=centered.description,
        ),
    )
    monkeypatch.setitem(
        REMOTE_DATASETS,
        "bad_checksum",
        RemoteFileMetadata(
            name="bad_checksum",
            filename=filename,
            url=url,
            checksum="bad!",
            description=centered.description,
        ),
    )
    UnknownFileMetaData = namedtuple(
        "UnknownFileMetaData", ["filename", "url", "checksum", "description"]
    )
    monkeypatch.setitem(
        REMOTE_DATASETS,
        "test_unknown",
        UnknownFileMetaData(
            filename=filename,
            url=url,
            checksum="9ae00c83654b3f061d32c882ec0a270d10838fa36515ecb162b89a290e014849",
            description="Test bad REMOTE_DATASET",
        ),
    )


@netcdf_nightlies_skip
@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed")
def test_load_local_arviz_data():
    idata = load_arviz_data("centered_eight")
    assert isinstance(idata, DataTree)
    assert set(idata.observed_data.obs.coords["school"].values) == {
        "Hotchkiss",
        "Mt. Hermon",
        "Choate",
        "Deerfield",
        "Phillips Andover",
        "St. Paul's",
        "Lawrenceville",
        "Phillips Exeter",
    }
    assert idata.posterior["theta"].dims == ("chain", "draw", "school")


@netcdf_nightlies_skip
def test_clear_data_home():
    resource = REMOTE_DATASETS["test_remote"]
    assert not os.path.exists(resource.filename)
    load_arviz_data("test_remote")
    assert os.path.exists(resource.filename)
    clear_data_home(data_home=os.path.dirname(resource.filename))
    assert not os.path.exists(resource.filename)


@netcdf_nightlies_skip
def test_load_remote_arviz_data():
    assert load_arviz_data("test_remote")


def test_bad_checksum():
    with pytest.raises(IOError):
        load_arviz_data("bad_checksum")


def test_missing_dataset():
    with pytest.raises(ValueError):
        load_arviz_data("does not exist")


def test_list_datasets():
    dataset_string = list_datasets()
    # make sure all the names of the data sets are in the dataset description
    for key in (
        "centered_eight",
        "non_centered_eight",
        "test_remote",
        "bad_checksum",
        "test_unknown",
    ):
        assert key in dataset_string


def test_dims_coords():
    shape = 4, 20, 5
    var_name = "x"
    dims, coords = generate_dims_coords(shape, var_name)
    assert "x_dim_0" in dims
    assert "x_dim_1" in dims
    assert "x_dim_2" in dims
    assert len(coords["x_dim_0"]) == 4
    assert len(coords["x_dim_1"]) == 20
    assert len(coords["x_dim_2"]) == 5


def test_dims_coords_extra_dims():
    shape = 4, 20
    var_name = "x"
    with pytest.raises(ValueError, match="more dims"):
        generate_dims_coords(shape, var_name, dims=["xx", "xy", "xz"])


@pytest.mark.parametrize("shape", [(4, 20), (4, 20, 1)])
def test_dims_coords_skip_event_dims(shape):
    coords_raw: dict[Hashable, ArrayLike] = {
        "x": np.arange(4),
        "y": np.arange(20),
        "z": np.arange(5),
    }
    dims, coords = generate_dims_coords(
        shape, "name", dims=["x", "y", "z"], coords=coords_raw, skip_event_dims=True
    )
    assert "x" in dims
    assert "y" in dims
    assert "z" not in dims
    assert len(coords["x"]) == 4
    assert len(coords["y"]) == 20
    assert "z" not in coords


@pytest.mark.parametrize("dims", [None, ["chain", "draw"], ["chain", "draw", None]])
def test_autonaming_single_dim(dims):
    sample_dims = ["chain", "draw"]
    da_sample_dims = ndarray_to_dataarray(
        np.empty((4, 500, 7)),
        "a",
        dims=dims,
        sample_dims=sample_dims,
    )
    da_no_sample_dims = ndarray_to_dataarray(
        np.empty(7),
        "a",
        dims=[dim for dim in dims if dim not in sample_dims]
        if isinstance(dims, Iterable)
        else dims,
        sample_dims=[],
    )
    assert list(da_sample_dims.dims) == ["chain", "draw", "a_dim_0"]
    assert list(da_no_sample_dims.dims) == [
        dim for dim in da_sample_dims.dims if dim not in sample_dims
    ]


@pytest.mark.parametrize(
    "args",
    [
        (None, (4, 500, 7, 3), ["chain", "draw", "a_dim_0", "a_dim_1"]),
        (["chain", "draw"], (4, 500, 7, 3), ["chain", "draw", "a_dim_0", "a_dim_1"]),
        (["chain", "draw", None], (4, 500, 7, 3), ["chain", "draw", "a_dim_0", "a_dim_1"]),
        (["chain", "draw", None, "group"], (4, 500, 7, 3), ["chain", "draw", "a_dim_0", "group"]),
        (["group", None, "chain"], (500, 3, 7, 4), ["draw", "group", "a_dim_0", "chain"]),
        ([None, "group", "chain"], (500, 7, 3, 4), ["draw", "a_dim_0", "group", "chain"]),
        ([None, "draw", "chain", None], (7, 500, 4, 3), ["a_dim_0", "draw", "chain", "a_dim_1"]),
    ],
)
def test_autonaming_multiple_dims(args):
    dims, shape, out_dims = args
    sample_dims = ["chain", "draw"]
    da_sample_dims = ndarray_to_dataarray(
        np.empty(shape),
        "a",
        dims=dims,
        sample_dims=sample_dims,
    )
    da_no_sample_dims = ndarray_to_dataarray(
        np.empty((7, 3)),
        "a",
        dims=[dim for dim in dims if dim not in sample_dims]
        if isinstance(dims, Iterable)
        else dims,
        sample_dims=[],
    )
    assert list(da_sample_dims.dims) == out_dims
    assert list(da_no_sample_dims.dims) == [
        dim for dim in da_sample_dims.dims if dim not in sample_dims
    ]


def test_make_attrs():
    extra_attrs = {"key": "Value"}
    attrs = make_attrs(attrs=extra_attrs)
    assert "key" in attrs
    assert attrs["key"] == "Value"
    assert "created_at" in attrs
    assert "creation_library" in attrs


def test_dict_to_dataset():
    rng = np.random.default_rng()
    datadict: dict[Hashable, ArrayLike] = {
        "a": rng.normal(size=(1, 100)),
        "b": rng.normal(size=(1, 100, 10)),
    }
    dataset = dict_to_dataset(datadict, coords={"c": np.arange(10)}, dims={"b": ["c"]})
    assert set(dataset.data_vars) == {"a", "b"}
    assert set(dataset.coords) == {"chain", "draw", "c"}

    assert set(dataset.a.coords) == {"chain", "draw"}
    assert set(dataset.b.coords) == {"chain", "draw", "c"}


def test_dict_to_dataset_event_dims_error():
    rng = np.random.default_rng()
    datadict: dict[Hashable, ArrayLike] = {"a": rng.normal(size=(1, 100, 10))}
    coords: dict[Hashable, ArrayLike] = {"b": np.arange(10), "c": ["x", "y", "z"]}
    msg = "more dims"
    with pytest.raises(ValueError, match=msg):
        dict_to_dataset(datadict, coords=coords, dims={"a": ["b", "c"]})


def test_dict_to_dataset_with_tuple_coord():
    rng = np.random.default_rng()
    datadict: dict[Hashable, ArrayLike] = {
        "a": rng.normal(size=(1, 100)),
        "b": rng.normal(size=(1, 100, 10)),
    }
    with pytest.raises(TypeError, match="Could not convert tuple"):
        dict_to_dataset(datadict, coords={"c": tuple(range(10))}, dims={"b": ["c"]})


@pytest.mark.parametrize(
    "args",
    [
        (["chain", "draw"], (4, 10)),
        (["sample"], (10,)),
        (["chain", "draw", "pred_id"], (4, 10, 3)),
    ],
)
def test_ndarray_to_dataarray(args):
    dims, shape = args
    rng = np.random.default_rng(3)
    ary = rng.normal(size=shape)
    with_dims = ndarray_to_dataarray(ary, "x", dims=dims, sample_dims=[])
    with_sample_dims = ndarray_to_dataarray(ary, "x", dims=[], sample_dims=dims)
    assert_allclose(with_sample_dims, with_dims)


@pytest.mark.parametrize("mode", ["scalar", "0d array"])
def test_ndarray_to_dataarray_scalar(mode):
    ary: ArrayLike | Number
    if mode == "scalar":
        ary = 2
    else:
        ary = np.array(2)
    da = ndarray_to_dataarray(ary, "x", dims=[], sample_dims=[])
    assert not da.dims
    assert da.item() == 2
