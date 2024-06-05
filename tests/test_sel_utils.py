# pylint: disable=redefined-outer-name
import numpy as np
import pytest
import xarray as xr
from arviz_base import dict_to_dataset, xarray_sel_iter, xarray_to_ndarray


@pytest.fixture(scope="function")
def sample_dataset():
    mu = np.arange(1, 7).reshape(2, 3)
    tau = np.arange(7, 13).reshape(2, 3)

    chain = [0, 1]
    draws = [0, 1, 2]

    data = xr.Dataset(
        {"mu": (["chain", "draw"], mu), "tau": (["chain", "draw"], tau)},
        coords={"draw": draws, "chain": chain},
    )

    return mu, tau, data


@pytest.fixture(scope="function")
def double_index_dataarray():
    return (
        xr.DataArray(
            np.random.default_rng(2).normal(size=(2, 5, 7)),
            dims=["chain", "draw", "obs_dim"],
            coords={
                "chain": [1, 2],
                "obs_id": ("obs_dim", np.arange(7)),
                "label_id": ("obs_dim", list("babacbc")),
            },
            name="sample",
        )
        .set_xindex("obs_id")
        .set_xindex("label_id")
    )


def test_dataset_to_numpy_combined(sample_dataset):
    mu, tau, data = sample_dataset
    var_names, data = xarray_to_ndarray(data, combined=True)

    assert len(var_names) == 2
    assert (data[var_names.index("mu")] == mu.reshape(1, 6)).all()
    assert (data[var_names.index("tau")] == tau.reshape(1, 6)).all()


def test_xarray_sel_iter_ordering():
    """Assert that coordinate names stay the provided order"""
    coords = list("dcba")
    rng = np.random.default_rng()
    data = dict_to_dataset(
        {"x": rng.normal(size=(1, 100, len(coords)))},
        coords={"in_order": coords},
        dims={"x": ["in_order"]},
    )

    coord_names = [sel["in_order"] for _, sel, _ in xarray_sel_iter(data)]
    assert coord_names == coords


def test_xarray_sel_iter_ordering_combined(sample_dataset):
    """Assert that varname order stays consistent when chains are combined"""
    _, _, data = sample_dataset
    var_names = [var for (var, _, _) in xarray_sel_iter(data, var_names=None, combined=True)]
    assert set(var_names) == {"mu", "tau"}


def test_xarray_sel_iter_ordering_uncombined(sample_dataset):
    """Assert that varname order stays consistent when chains are not combined"""
    _, _, data = sample_dataset
    var_names = [
        (var, selection)
        for (var, selection, _) in xarray_sel_iter(data, var_names=None, combined=False)
    ]

    assert len(var_names) == 4
    for var_name in var_names:
        assert var_name in [
            ("mu", {"chain": 0}),
            ("mu", {"chain": 1}),
            ("tau", {"chain": 0}),
            ("tau", {"chain": 1}),
        ]


def test_xarray_sel_data_array(sample_dataset):
    """Assert that varname order stays consistent when chains are combined

    Touches code that is hard to reach.
    """
    _, _, data = sample_dataset
    var_names = [var for (var, _, _) in xarray_sel_iter(data.mu, var_names=None, combined=True)]
    assert set(var_names) == {"mu"}


def test_xarray_sel_xindexes_unique(double_index_dataarray):
    out = list(
        xarray_sel_iter(double_index_dataarray, combined=False, dim_to_idx={"obs_dim": "obs_id"})
    )

    assert len(out) == 14
    assert out[0] == ("sample", {"chain": 1, "obs_id": 0}, {"chain": 0, "obs_dim": 0})


def test_xarray_sel_xindexes_nonunique(double_index_dataarray):
    out = list(
        xarray_sel_iter(double_index_dataarray, combined=False, dim_to_idx={"obs_dim": "label_id"})
    )

    assert len(out) == 6
    assert out[0][0] == "sample"
    assert out[0][1] == {"chain": 1, "label_id": "b"}
    assert "obs_dim" in out[0][2]
    idx_where = out[0][2]["obs_dim"]
    assert len(idx_where) == 3
    assert np.all(np.isin(idx_where, [0, 2, 5]))
