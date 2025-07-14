import xarray.testing as xrt
from xarray import Dataset, DataTree

from arviz_base import get_unconstrained_samples


def test_no_transform_funcs_no_unconstrained(centered_eight):
    # idata with only posterior group
    idata = get_unconstrained_samples(centered_eight, transform_funcs=None)

    # idata has a "unconstrained_posterior" group
    assert "unconstrained_posterior" in idata

    # unconstrained_posterior should equal original posterior
    ds_uc = idata["unconstrained_posterior"]

    xrt.assert_equal(ds_uc, centered_eight.posterior)


def test_with_transform_funcs_no_unconstrained(centered_eight):
    # transform tau
    funcs = {"tau": lambda arr: 2 * arr}

    idata = get_unconstrained_samples(centered_eight, transform_funcs=funcs)

    ds_post = centered_eight["posterior"]
    ds_uc = idata["unconstrained_posterior"]
    # expected values of tau
    expected_tau = 2 * ds_post["tau"]
    xrt.assert_equal(ds_uc["tau"], expected_tau)
    # mu and theta unchanged
    xrt.assert_equal(ds_uc["mu"], ds_post["mu"])
    xrt.assert_equal(ds_uc["theta"], ds_post["theta"])


def test_no_transform_funcs_with_unconstrained(centered_eight):
    # create an unconstrained_posterior with only tau
    uc = Dataset({"tau": 2 * centered_eight["posterior"]["tau"]})
    idata_with_uc = DataTree.from_dict(
        {"posterior": centered_eight["posterior"], "unconstrained_posterior": uc}
    )
    out = get_unconstrained_samples(idata_with_uc, transform_funcs=None)

    ds_uc = out["unconstrained_posterior"]
    # tau taken from unconstrained_posterior
    xrt.assert_equal(ds_uc["tau"], uc["tau"])
    # mu taken from posterior
    xrt.assert_equal(ds_uc["mu"], centered_eight.posterior["mu"])


def test_with_transform_funcs_with_unconstrained(centered_eight):
    # transform mu by subtracting 3
    funcs = {"mu": lambda arr: arr - 3}
    uc = Dataset({"tau": 2 * centered_eight["posterior"]["tau"]})
    idata_with_uc = DataTree.from_dict(
        {"posterior": centered_eight["posterior"], "unconstrained_posterior": uc}
    )
    out = get_unconstrained_samples(idata_with_uc, transform_funcs=funcs)

    ds_uc = out["unconstrained_posterior"]
    # tau taken from unconstrained_posterior
    xrt.assert_equal(ds_uc["tau"], uc["tau"])
    # mu transformed
    # expected values of mu
    expected_mu = centered_eight["posterior"]["mu"] - 3
    xrt.assert_equal(ds_uc["mu"], expected_mu)


def test_with_transform_custom_groups(centered_eight):
    # transform y
    funcs = {"y": lambda arr: arr - 3}

    idata = get_unconstrained_samples(
        centered_eight,
        transform_funcs=funcs,
        group="posterior_predictive",
    )

    # return group is only group in idata
    assert ("/", "/unconstrained_posterior_predictive") == idata.groups

    ds_out = idata["unconstrained_posterior_predictive"]

    # y transformed
    # expected values of y
    expected_y = centered_eight["posterior_predictive"]["y"] - 3
    xrt.assert_equal(ds_out["y"], expected_y)
