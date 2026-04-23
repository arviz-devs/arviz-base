# pylint: disable=no-member, no-self-use, invalid-name, redefined-outer-name
"""Tests for from_zarr and from_netcdf top-level I/O functions."""

import os

import numpy as np
import pytest
import xarray as xr

import arviz_base as az
from arviz_base import from_netcdf, from_zarr

zarr = pytest.importorskip("zarr")

netcdf_nightlies_skip = pytest.mark.skipif(
    os.environ.get("NIGHTLIES", "FALSE") == "TRUE",
    reason="Skip netcdf4 dependent tests from nightlies as it generally takes longer to update.",
)


@pytest.fixture
def sample_datatree():
    """A minimal DataTree with posterior and observed_data groups."""
    rng = np.random.default_rng(0)
    posterior = xr.Dataset(
        {
            "alpha": (["chain", "draw"], rng.normal(size=(2, 50))),
            "beta": (["chain", "draw", "obs_dim"], rng.normal(size=(2, 50, 3))),
        },
        attrs={"created_by": "test", "inference_library": "pymc"},
    )
    observed_data = xr.Dataset(
        {"y": (["obs_dim"], rng.normal(size=3))},
        attrs={"description": "observed"},
    )
    return xr.DataTree.from_dict({"/posterior": posterior, "/observed_data": observed_data})


# ---------------------------------------------------------------------------
# from_zarr tests
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_zarr_roundtrip(sample_datatree, tmp_path):
    """Writing with DataTree.to_zarr and reading back with from_zarr is lossless."""
    store = str(tmp_path / "idata.zarr")
    sample_datatree.to_zarr(store)
    result = from_zarr(store)
    assert set(result.children) == set(sample_datatree.children)
    xr.testing.assert_identical(
        result["posterior"].to_dataset(), sample_datatree["posterior"].to_dataset()
    )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_zarr_preserves_attrs(sample_datatree, tmp_path):
    """Group-level attrs survive the zarr roundtrip."""
    store = str(tmp_path / "idata_attrs.zarr")
    sample_datatree.to_zarr(store)
    result = from_zarr(store)
    assert result["posterior"].attrs == sample_datatree["posterior"].attrs
    assert result["observed_data"].attrs == sample_datatree["observed_data"].attrs


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_from_zarr_lazy(sample_datatree, tmp_path):
    """from_zarr returns a DataTree whose data arrays are lazily loaded (dask-backed)."""
    pytest.importorskip("dask")
    store = str(tmp_path / "idata_lazy.zarr")
    sample_datatree.to_zarr(store)
    result = from_zarr(store, chunks={})
    alpha = result["posterior"].ds["alpha"]
    assert alpha.chunks is not None, "Expected lazy (chunked) array after from_zarr with chunks={}"


def test_from_zarr_is_partial():
    """from_zarr is a functools.partial of open_datatree with engine='zarr'."""
    import functools

    from xarray import open_datatree

    assert isinstance(from_zarr, functools.partial)
    assert from_zarr.func is open_datatree
    assert from_zarr.keywords.get("engine") == "zarr"


# ---------------------------------------------------------------------------
# from_netcdf tests
# ---------------------------------------------------------------------------


@netcdf_nightlies_skip
def test_from_netcdf_roundtrip(sample_datatree, tmp_path):
    """Writing with DataTree.to_netcdf and reading back with from_netcdf is lossless."""
    path = str(tmp_path / "idata.nc")
    sample_datatree.to_netcdf(path)
    result = from_netcdf(path)
    assert set(result.children) == set(sample_datatree.children)
    xr.testing.assert_identical(
        result["posterior"].to_dataset(), sample_datatree["posterior"].to_dataset()
    )


@netcdf_nightlies_skip
def test_from_netcdf_preserves_attrs(sample_datatree, tmp_path):
    """Group-level attrs survive the netcdf roundtrip."""
    path = str(tmp_path / "idata_attrs.nc")
    sample_datatree.to_netcdf(path)
    result = from_netcdf(path)
    assert result["posterior"].attrs == sample_datatree["posterior"].attrs


def test_from_netcdf_is_open_datatree():
    """from_netcdf is exactly open_datatree (identity alias)."""
    from xarray import open_datatree

    assert from_netcdf is open_datatree


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_from_zarr_in_all():
    assert "from_zarr" in az.__all__


def test_from_netcdf_in_all():
    assert "from_netcdf" in az.__all__
