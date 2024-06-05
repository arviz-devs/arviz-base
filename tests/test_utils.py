# pylint: disable=redefined-outer-name
import numpy as np
import pytest
from arviz_base import dict_to_dataset
from arviz_base.utils import _subset_list, _var_names

from .helpers import ExampleRandomVariable  # pylint: disable=unused-import


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng()
    samples = rng.normal(size=(2, 10))
    dataset = dict_to_dataset(
        {
            "mu": samples,  # pylint: disable=invalid-unary-operand-type
            "theta": samples,
            "tau": samples,  # pylint: disable=invalid-unary-operand-type
        }
    )
    return dataset


@pytest.mark.parametrize(
    "var_names_expected",
    [
        ("mu", ["mu"]),
        (None, None),
        (["mu", "tau"], ["mu", "tau"]),
        ("~mu", ["theta", "tau"]),
        (["~mu"], ["theta", "tau"]),
    ],
)
def test_var_names(var_names_expected, data):
    """Test var_name handling"""
    var_names, expected = var_names_expected
    assert _var_names(var_names, data) == expected


def test_var_names_warning():
    """Test confusing var_name handling"""
    rng = np.random.default_rng()
    ds = dict_to_dataset(
        {
            "~mu": rng.normal(size=(2, 10)),
            "mu": -rng.normal(size=(2, 10)),  # pylint: disable=invalid-unary-operand-type
            "theta": rng.normal(size=(2, 10, 8)),
        }
    )
    var_names = expected = ["~mu"]
    with pytest.warns(UserWarning):
        assert _var_names(var_names, ds) == expected


def test_var_names_key_error(data):
    with pytest.raises(KeyError, match="bad_var_name"):
        _var_names(("theta", "tau", "bad_var_name"), data)


@pytest.mark.parametrize(
    "var_args",
    [
        (["ta"], ["beta1", "beta2", "theta"], "like"),
        (["~beta"], ["phi", "theta"], "like"),
        (["beta[0-9]+"], ["beta1", "beta2"], "regex"),
        (["^p"], ["phi"], "regex"),
        (["~^t"], ["beta1", "beta2", "phi"], "regex"),
    ],
)
def test_var_names_filter_multiple_input(var_args):
    rng = np.random.default_rng()
    samples = rng.normal(size=(1, 10))
    data1 = dict_to_dataset({"beta1": samples, "beta2": samples, "phi": samples})
    data2 = dict_to_dataset({"beta1": samples, "beta2": samples, "theta": samples})
    data = [data1, data2]
    var_names, expected, filter_vars = var_args
    assert _var_names(var_names, data, filter_vars) == expected


@pytest.mark.parametrize(
    "var_args",
    [
        (["alpha", "beta"], ["alpha", "beta1", "beta2"], "like"),
        (["~beta"], ["alpha", "p1", "p2", "phi", "theta", "theta_t"], "like"),
        (["theta"], ["theta", "theta_t"], "like"),
        (["~theta"], ["alpha", "beta1", "beta2", "p1", "p2", "phi"], "like"),
        (["p"], ["alpha", "p1", "p2", "phi"], "like"),
        (["~p"], ["beta1", "beta2", "theta", "theta_t"], "like"),
        (["^bet"], ["beta1", "beta2"], "regex"),
        (["^p"], ["p1", "p2", "phi"], "regex"),
        (["~^p"], ["alpha", "beta1", "beta2", "theta", "theta_t"], "regex"),
        (["p[0-9]+"], ["p1", "p2"], "regex"),
        (["~p[0-9]+"], ["alpha", "beta1", "beta2", "phi", "theta", "theta_t"], "regex"),
    ],
)
def test_var_names_filter(var_args):
    """Test var_names filter with partial naming or regular expressions."""
    rng = np.random.default_rng()
    samples = rng.normal(size=(1, 10))
    data = dict_to_dataset(
        {
            "alpha": samples,
            "beta1": samples,
            "beta2": samples,
            "p1": samples,
            "p2": samples,
            "phi": samples,
            "theta": samples,
            "theta_t": samples,
        }
    )
    var_names, expected, filter_vars = var_args
    assert _var_names(var_names, data, filter_vars) == expected


def test_nonstring_var_names():
    """Check that non-string variables are preserved"""
    mu = ExampleRandomVariable("mu")
    rng = np.random.default_rng()
    samples = rng.normal(size=(1, 10))
    data = dict_to_dataset({mu: samples})
    assert _var_names([mu], data) == [mu]


def test_var_names_filter_invalid_argument():
    """Check invalid argument raises."""
    rng = np.random.default_rng()
    samples = rng.normal(size=(1, 10))
    data = dict_to_dataset({"alpha": samples})
    msg = r"^\'filter_vars\' can only be None, \'like\', or \'regex\', got: 'foo'$"
    with pytest.raises(ValueError, match=msg):
        assert _var_names(["alpha"], data, filter_vars="foo")


def test_subset_list_negation_not_found():
    """Check there is a warning if negation pattern is ignored"""
    names = ["mu", "theta"]
    with pytest.warns(UserWarning, match=".+not.+found.+"):
        assert _subset_list("~tau", names) == names
