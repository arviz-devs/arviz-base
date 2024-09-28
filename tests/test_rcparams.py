# pylint: disable=redefined-outer-name
import os

import numpy as np
import pytest
from xarray.testing import assert_allclose

from arviz_base import ndarray_to_dataarray
from arviz_base.rcparams import (
    _make_validate_choice,
    _make_validate_choice_regex,
    _validate_float_or_none,
    _validate_positive_int_or_none,
    _validate_probability,
    _validate_stats_module,
    make_iterable_validator,
    rc_context,
    rcParams,
    read_rcfile,
)


### Test rcparams classes ###
def test_rc_context_dict():
    rcParams["data.http_protocol"] = "https"
    with rc_context(rc={"data.http_protocol": "http"}):
        assert rcParams["data.http_protocol"] == "http"
    assert rcParams["data.http_protocol"] == "https"


def test_rc_context_file():
    path = os.path.dirname(os.path.abspath(__file__))
    rcParams["stats.point_estimate"] = "mean"
    with rc_context(fname=os.path.join(path, "valid.rcparams")):
        assert rcParams["stats.point_estimate"] == "median"
    assert rcParams["stats.point_estimate"] == "mean"


def test_bad_rc_file():
    """Test bad value raises error."""
    path = os.path.dirname(os.path.abspath(__file__))
    with pytest.raises(ValueError, match="Bad val "):
        read_rcfile(os.path.join(path, "bad.rcparams"))


def test_warning_rc_file(caplog):
    """Test invalid lines and duplicated keys log warnings and bad value raises error."""
    path = os.path.dirname(os.path.abspath(__file__))
    read_rcfile(os.path.join(path, "valid.rcparams"))
    records = caplog.records
    assert len(records) == 1
    assert records[0].levelname == "WARNING"
    assert "Duplicate key" in caplog.text


def test_bad_key():
    """Test the error when using unexistent keys in rcParams is correct."""
    with pytest.raises(KeyError, match="bad_key is not a valid rc"):
        rcParams["bad_key"] = "nothing"


def test_del_key_error():
    """Check that rcParams keys cannot be deleted."""
    with pytest.raises(TypeError, match="keys cannot be deleted"):
        del rcParams["data.http_protocol"]


def test_clear_error():
    """Check that rcParams cannot be cleared."""
    with pytest.raises(TypeError, match="keys cannot be deleted"):
        rcParams.clear()


def test_pop_error():
    """Check rcParams pop error."""
    with pytest.raises(TypeError, match=r"keys cannot be deleted.*get\(key\)"):
        rcParams.pop("data.http_protocol")


def test_popitem_error():
    """Check rcParams popitem error."""
    with pytest.raises(TypeError, match=r"keys cannot be deleted.*get\(key\)"):
        rcParams.popitem()


def test_setdefaults_error():
    """Check rcParams popitem error."""
    with pytest.raises(TypeError, match="Use arvizrc"):
        rcParams.setdefault("data.http_protocol", "https")


def test_rcparams_find_all():
    data_rcparams = rcParams.find_all("data")
    assert len(data_rcparams)


def test_rcparams_repr_str():
    """Check both repr and str print all keys."""
    repr_str = repr(rcParams)
    str_str = str(rcParams)
    assert repr_str.startswith("RcParams")
    for string in (repr_str, str_str):
        assert all(key in string for key in rcParams.keys())


### Test validation functions ###
@pytest.mark.parametrize("param", ["data.http_protocol", "stats.information_criterion"])
def test_choice_bad_values(param):
    """Test error messages are correct for rcParams validated with _make_validate_choice."""
    msg = "{}: bad_value is not one of".format(param.replace(".", r"\."))
    with pytest.raises(ValueError, match=msg):
        rcParams[param] = "bad_value"


@pytest.mark.parametrize("allow_none", (True, False))
@pytest.mark.parametrize("typeof", (str, int))
@pytest.mark.parametrize("args", [("not one", 10), (False, None), (False, 4)])
def test_make_validate_choice(args, allow_none, typeof):
    accepted_values = set(typeof(value) for value in (0, 1, 4, 6))
    validate_choice = _make_validate_choice(accepted_values, allow_none=allow_none, typeof=typeof)
    raise_error, value = args
    if value is None and not allow_none:
        raise_error = "not one of" if typeof is str else "Could not convert"
    if raise_error:
        with pytest.raises(ValueError, match=raise_error):
            validate_choice(value)
    else:
        value = validate_choice(value)
        assert value in accepted_values or value is None


@pytest.mark.parametrize("allow_none", (True, False))
@pytest.mark.parametrize(
    "args",
    [
        (False, None),
        (False, "row"),
        (False, "54row"),
        (False, "4column"),
        ("or in regex", "square"),
    ],
)
def test_make_validate_choice_regex(args, allow_none):
    accepted_values = {"row", "column"}
    accepted_values_regex = {r"\d*row", r"\d*column"}
    validate_choice = _make_validate_choice_regex(
        accepted_values, accepted_values_regex, allow_none=allow_none
    )
    raise_error, value = args
    if value is None and not allow_none:
        raise_error = "or in regex"
    if raise_error:
        with pytest.raises(ValueError, match=raise_error):
            validate_choice(value)
    else:
        value_result = validate_choice(value)
        assert value == value_result


@pytest.mark.parametrize("allow_none", (True, False))
@pytest.mark.parametrize("allow_auto", (True, False))
@pytest.mark.parametrize("value", [(1, 2), "auto", None, "(1, 4)"])
def test_make_iterable_validator_none_auto(value, allow_auto, allow_none):
    scalar_validator = _validate_float_or_none
    validate_iterable = make_iterable_validator(
        scalar_validator, allow_auto=allow_auto, allow_none=allow_none
    )
    raise_error = False
    if value is None and not allow_none:
        raise_error = "Only ordered iterable"
    if value == "auto" and not allow_auto:
        raise_error = "Could not convert"
    if raise_error:
        with pytest.raises(ValueError, match=raise_error):
            validate_iterable(value)
    else:
        value = validate_iterable(value)
        assert np.iterable(value) or value is None or value == "auto"


@pytest.mark.parametrize("length", (2, None))
@pytest.mark.parametrize("value", [(1, 5), (1, 3, 5), "(3, 4, 5)"])
def test_make_iterable_validator_length(value, length):
    scalar_validator = _validate_float_or_none
    validate_iterable = make_iterable_validator(scalar_validator, length=length)
    raise_error = False
    if length is not None and len(value) != length:
        raise_error = "Iterable must be of length"
    if raise_error:
        with pytest.raises(ValueError, match=raise_error):
            validate_iterable(value)
    else:
        value = validate_iterable(value)
        assert np.iterable(value)


@pytest.mark.parametrize(
    "args",
    [
        ("Only ordered iterable", set(["a", "b", "c"])),
        ("Could not convert", "johndoe"),
        ("Only ordered iterable", 15),
    ],
)
def test_make_iterable_validator_illegal(args):
    scalar_validator = _validate_float_or_none
    validate_iterable = make_iterable_validator(scalar_validator)
    raise_error, value = args
    with pytest.raises(ValueError, match=raise_error):
        validate_iterable(value)


@pytest.mark.parametrize(
    "args",
    [
        ("Only positive", -1),
        ("Could not convert", "1.3"),
        (False, "2"),
        (False, None),
        (False, 1),
    ],
)
def test_validate_positive_int_or_none(args):
    raise_error, value = args
    if raise_error:
        with pytest.raises(ValueError, match=raise_error):
            _validate_positive_int_or_none(value)
    else:
        value = _validate_positive_int_or_none(value)
        assert isinstance(value, int) or value is None


@pytest.mark.parametrize(
    "args",
    [
        ("Only.+between 0 and 1", -1),
        ("Only.+between 0 and 1", "1.3"),
        ("not convert to float", "word"),
        (False, "0.6"),
        (False, 0),
        (False, 1),
    ],
)
def test_validate_probability(args):
    raise_error, value = args
    if raise_error:
        with pytest.raises(ValueError, match=raise_error):
            _validate_probability(value)
    else:
        value = _validate_probability(value)
        assert isinstance(value, float)


# pylint: disable=no-self-use
class MockGoodStats:
    def eti(self):
        return

    def rhat(self):
        return


# pylint: disable=no-self-use
class MockBadStats:
    eti = False

    def rhat(self):
        return


@pytest.mark.parametrize(
    "args",
    [
        (False, "base"),
        (False, MockGoodStats()),
        ("Only.+statistical functions", MockBadStats()),
    ],
)
def test_validate_stats_module(args):
    raise_error, value = args
    if raise_error:
        with pytest.raises(ValueError, match=raise_error):
            _validate_stats_module(value)
    else:
        validated = _validate_stats_module(value)
        assert value is validated


## Some simple integration checks with rcparams
def test_sample_dims():
    rng = np.random.default_rng(3)
    ary = rng.normal(size=(4, 10, 3))
    da1 = ndarray_to_dataarray(ary[:, :, 0], "x", dims=[])
    assert "chain" in da1.dims
    assert "draw" in da1.dims
    assert da1.shape == (4, 10)
    with rc_context(rc={"data.sample_dims": ["chain", "draw", "pred_id"]}):
        da2 = ndarray_to_dataarray(ary, "x", dims=[])
        assert "chain" in da2.dims
        assert "draw" in da2.dims
        assert "pred_id" in da2.dims
        assert da2.shape == (4, 10, 3)
        assert_allclose(da1, da2.sel(pred_id=0, drop=True))
