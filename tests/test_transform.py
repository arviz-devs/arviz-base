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
