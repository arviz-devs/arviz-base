"""Test validation functions."""

from collections.abc import Mapping
from typing import Any, Literal

import pytest
import xarray as xr

from arviz_base import rcParams
from arviz_base.validate import (
    validate_ci_prob,
    validate_dict_argument,
    validate_dims_chain_draw_axis,
    validate_or_use_rcparam,
    validate_prob,
    validate_sample_dims,
)


def test_validate_sample_dims_none():
    result = validate_sample_dims(None)
    expected = rcParams["data.sample_dims"]
    assert result == list(expected)


def test_validate_sample_dims_string():
    result = validate_sample_dims("chain")
    assert result == ["chain"]


def test_validate_sample_dims_list():
    result = validate_sample_dims(["chain", "draw"])
    assert result == ["chain", "draw"]


def test_validate_sample_dims_tuple():
    result = validate_sample_dims(("chain", "draw"))
    assert result == ["chain", "draw"]


def test_validate_sample_dims_single_element():
    result = validate_sample_dims(["sample"])
    assert result == ["sample"]


def test_validate_sample_dims_attr():
    ds = xr.Dataset({"var": (("sample",), [0, 1, 2, 3, 4])}, attrs={"sample_dims": ["sample"]})
    result = validate_sample_dims(None, data=ds)
    assert result == ["sample"]


def test_validate_sample_dims_error():
    ds = xr.Dataset({"var": (("sample",), [0, 1, 2, 3, 4])}, attrs={"sample_dims": ["sample"]})
    with pytest.raises(ValueError, match="Invalid value for sample_dims"):
        validate_sample_dims("draw", data=ds)


def test_validate_dims_chain_draw_axis_two_dims():
    dims, chain_axis, draw_axis = validate_dims_chain_draw_axis(["chain", "draw"])
    assert dims == ["chain", "draw"]
    assert chain_axis == -2
    assert draw_axis == -1


def test_validate_dims_chain_draw_axis_one_dim():
    dims, chain_axis, draw_axis = validate_dims_chain_draw_axis(["draw"])
    assert dims == ["draw"]
    assert chain_axis is None
    assert draw_axis == -1


def test_validate_dims_chain_draw_axis_none():
    dims, chain_axis, draw_axis = validate_dims_chain_draw_axis(None)
    expected = rcParams["data.sample_dims"]
    assert dims == list(expected)
    if len(dims) == 2:
        assert chain_axis == -2
        assert draw_axis == -1
    else:
        assert chain_axis is None
        assert draw_axis == -1


def test_validate_dims_chain_draw_axis_string():
    dims, chain_axis, draw_axis = validate_dims_chain_draw_axis("draw")
    assert dims == ["draw"]
    assert chain_axis is None
    assert draw_axis == -1


def test_validate_dims_chain_draw_axis_invalid_length():
    with pytest.raises(ValueError, match="dims can only have 1 or 2 elements"):
        validate_dims_chain_draw_axis(["chain", "draw", "extra"])


def test_validate_ci_prob_none():
    result = validate_ci_prob(None)
    expected = rcParams["stats.ci_prob"]
    assert result == expected


def test_validate_ci_prob_valid():
    result = validate_ci_prob(0.95)
    assert result == 0.95


def test_validate_ci_prob_boundary_upper():
    result = validate_ci_prob(1.0)
    assert result == 1.0


def test_validate_ci_prob_boundary_lower():
    result = validate_ci_prob(0.01)
    assert result == 0.01


def test_validate_prob_valid():
    result = validate_prob(0.5)
    assert result == 0.5


def test_validate_prob_boundary_upper():
    result = validate_prob(1.0)
    assert result == 1.0


def test_validate_prob_boundary_lower():
    result = validate_prob(0.001)
    assert result == 0.001


def test_validate_prob_invalid_too_high():
    with pytest.raises(ValueError, match="should be in the interval"):
        validate_prob(1.1)


def test_validate_prob_invalid_negative():
    with pytest.raises(ValueError, match="should be in the interval"):
        validate_prob(-0.1)


def test_validate_prob_invalid_zero():
    with pytest.raises(ValueError, match="should be in the interval"):
        validate_prob(0.0)


def test_validate_prob_allow_zero_true():
    result = validate_prob(0.0, allow_0=True)
    assert result == 0.0


def test_validate_prob_allow_zero_valid():
    result = validate_prob(0.5, allow_0=True)
    assert result == 0.5


def test_validate_prob_allow_zero_boundary():
    result = validate_prob(1.0, allow_0=True)
    assert result == 1.0


def test_validate_prob_allow_zero_invalid():
    with pytest.raises(ValueError, match="should be in the interval"):
        validate_prob(-0.1, allow_0=True)


def test_validate_ci_prob_invalid():
    with pytest.raises(ValueError, match="should be in the interval"):
        validate_ci_prob(1.5)


def test_validate_or_use_rcparam_default():
    result = validate_or_use_rcparam(None, "stats.ci_kind")
    assert result == rcParams["stats.ci_kind"]


def test_validate_or_use_rcparam_explicit():
    result = validate_or_use_rcparam("hdi", "stats.ci_kind")
    assert result == "hdi"


def test_validate_or_use_rcparam_error():
    with pytest.raises(ValueError):
        validate_or_use_rcparam("not_a_ci_type", "stats.ci_kind")


def plot_xyz(
    arg1: Mapping[Literal["key1", "key2"], Any], arg2: Mapping[Literal["key1", "key2", "key3"], Any]
):
    """Help test dict argument validation through type hints."""
    return (arg1, arg2)


def test_validate_dict_func():
    input_dict = {"key1": 1, "key2": 2, "key3": 3}
    result = validate_dict_argument(input_dict, (plot_xyz, "arg2"))
    assert result == input_dict

    with pytest.raises(ValueError, match=r"Found keys \['key3'"):
        validate_dict_argument(input_dict, (plot_xyz, "arg1"))


def test_validate_dict_explicit():
    input_dict = {"key1": 1, "key2": 2, "key3": 3}
    result = validate_dict_argument(input_dict, valid_keys=("key1", "key2", "key3"))
    assert result == input_dict

    with pytest.raises(ValueError, match=r"Found keys \['key3'"):
        validate_dict_argument(input_dict, valid_keys=("key1", "key2"))
