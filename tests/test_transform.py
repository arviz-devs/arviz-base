import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt

from arviz_base import get_unconstrained_samples


def test_no_transform_funcs_no_unconstrained(centered_eight):
    # idata with only posterior group
    idata = get_unconstrained_samples(centered_eight, transform_funcs=None)

    # it returns a datatree and return group is also a datatree
    assert isinstance(idata, xr.DataTree)

    # idata has a datatree "unconstrained_posterior" group
    assert "unconstrained_posterior" in idata
    assert isinstance(idata["unconstrained_posterior"], xr.DataTree)

    # unconstrained_posterior should equal original posterior
    dt_uc = idata["unconstrained_posterior"]

    xrt.assert_equal(dt_uc, centered_eight.posterior)


def test_with_transform_funcs_no_unconstrained(centered_eight):
    # transform tau
    funcs = {"tau": lambda arr: 2 * arr}

    idata = get_unconstrained_samples(centered_eight, transform_funcs=funcs)

    dt_post = centered_eight["posterior"]
    dt_uc = idata["unconstrained_posterior"]
    # expected values of tau
    expected_tau = 2 * dt_post["tau"]
    xrt.assert_equal(dt_uc["tau"], expected_tau)
    # mu and theta unchanged
    xrt.assert_equal(dt_uc["mu"], dt_post["mu"])
    xrt.assert_equal(dt_uc["theta"], dt_post["theta"])


def test_with_invalid_transform_funcs(centered_eight):
    funcs = {"tau": lambda arr: arr.to_dict()}

    with pytest.raises(TypeError, match="dict"):
        get_unconstrained_samples(centered_eight, transform_funcs=funcs)


def test_no_transform_funcs_with_unconstrained(centered_eight):
    # create an unconstrained_posterior with only tau
    uc = xr.Dataset({"tau": 2 * centered_eight["posterior"]["tau"]})
    idata_with_uc = xr.DataTree.from_dict(
        {"posterior": centered_eight["posterior"], "unconstrained_posterior": uc}
    )
    out = get_unconstrained_samples(idata_with_uc, transform_funcs=None)

    dt_uc = out["unconstrained_posterior"]
    # tau taken from unconstrained_posterior
    xrt.assert_equal(dt_uc["tau"], uc["tau"])
    # mu taken from posterior
    xrt.assert_equal(dt_uc["mu"], centered_eight.posterior["mu"])


def test_with_transform_funcs_with_unconstrained(centered_eight):
    # transform mu by subtracting 3
    funcs = {"mu": lambda arr: arr - 3}
    uc = xr.Dataset({"tau": 2 * centered_eight["posterior"]["tau"]})
    idata_with_uc = xr.DataTree.from_dict(
        {"posterior": centered_eight["posterior"], "unconstrained_posterior": uc}
    )
    out = get_unconstrained_samples(idata_with_uc, transform_funcs=funcs)

    dt_uc = out["unconstrained_posterior"]
    # tau taken from unconstrained_posterior
    xrt.assert_equal(dt_uc["tau"], uc["tau"])
    # mu transformed
    # expected values of mu
    expected_mu = centered_eight["posterior"]["mu"] - 3
    xrt.assert_equal(dt_uc["mu"], expected_mu)


def test_with_transform_custom_groups(centered_eight):
    # transform y
    funcs = {"y": lambda arr: arr - 3}

    idata = get_unconstrained_samples(
        centered_eight,
        transform_funcs=funcs,
        group="posterior_predictive",
    )

    # return group is only group in idata
    assert "/unconstrained_posterior_predictive" in idata.groups

    dt_out = idata["unconstrained_posterior_predictive"]

    # y transformed
    # expected values of y
    expected_y = centered_eight["posterior_predictive"]["y"] - 3
    xrt.assert_equal(dt_out["y"], expected_y)


def test_with_transform_return_dataset(centered_eight):
    # transform mu
    funcs = {"mu": lambda arr: arr - 3}

    ds_uc = get_unconstrained_samples(
        centered_eight,
        transform_funcs=funcs,
        return_dataset=True,
    )

    # it returns a datatree
    assert isinstance(ds_uc, xr.Dataset)

    # y transformed
    # expected values of mu
    expected_mu = centered_eight["posterior"]["mu"] - 3
    xrt.assert_equal(ds_uc["mu"], expected_mu)


def test_with_transform_filter_vars(centered_eight):
    # exclude theta
    idata = get_unconstrained_samples(
        centered_eight,
        var_names="~theta",
    )

    # it returns a datatree
    assert isinstance(idata, xr.DataTree)
    dt_out = idata["unconstrained_posterior"]
    assert "theta" not in dt_out
    assert "mu" in dt_out


def test_with_transform_dimensionality_changed(centered_eight):
    # transform theta
    def lagged_cumsum(da):
        core_dim = da.dims[-1]
        return da.cumsum(core_dim).isel({core_dim: slice(None, -1)})

    funcs = {"theta": lagged_cumsum}

    idata = get_unconstrained_samples(centered_eight, transform_funcs=funcs)

    dt_post = centered_eight["posterior"]
    dt_uc = idata["unconstrained_posterior"]
    # expected values of theta
    expected_theta = lagged_cumsum(dt_post["theta"])
    xrt.assert_equal(dt_uc["theta"], expected_theta)

    # assert dims and shape
    assert dt_uc["theta"].dims == dt_post["theta"].dims
    assert dt_uc["theta"].coords.keys() == dt_post["theta"].coords.keys()
    assert dt_uc["theta"].coords["school"].size == dt_post["theta"].coords["school"].size - 1
    assert dt_uc["theta"].shape[0:2] == dt_post["theta"].shape[0:2]
    assert dt_uc["theta"].shape[2] == dt_post["theta"].shape[2] - 1


def test_with_transform_change_dim_one_var(centered_eight):
    # transform mu
    def lagged_cumsum(da):
        core_dim = da.dims[-1]
        return da.cumsum(core_dim).isel({core_dim: slice(None, -1)})

    funcs = {"mu": lagged_cumsum}

    idata = get_unconstrained_samples(centered_eight, transform_funcs=funcs)

    dt_post = centered_eight["posterior"]
    dt_uc = idata["unconstrained_posterior"]
    # expected values of mu
    transformed_mu = lagged_cumsum(dt_post["mu"])
    # fill nan as dimension can't be removed
    filling_mu = xr.DataArray(np.array([np.nan]), coords={"draw": [9]}, dims="draw")
    expected_mu = xr.concat([transformed_mu, filling_mu], dim="draw")
    xrt.assert_equal(dt_uc["mu"], expected_mu)

    # assert dims and shape
    assert dt_uc.coords.equals(dt_post.coords)
    assert dt_uc["mu"].dims == dt_post["mu"].dims
    assert dt_uc["mu"].shape == dt_post["mu"].shape
