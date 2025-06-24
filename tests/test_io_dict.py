# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from arviz_base import from_dict
from arviz_base.testing import check_multiple_attrs


@pytest.fixture(scope="function")
def data(draws, chains):
    rng = np.random.default_rng()

    class Data:
        # fake 8-school output
        obj = {}
        for key, shape in {"mu": [], "tau": [], "eta": [8], "theta": [8]}.items():
            obj[key] = rng.normal(size=(chains, draws, *shape))

    return Data


def check_var_names_coords_dims(dataset):
    assert set(dataset.data_vars) == {"mu", "tau", "eta", "theta"}
    assert set(dataset.coords) == {"chain", "draw", "school"}


@pytest.mark.parametrize("save_warmup", (True, False))
def test_from_dict(data, eight_schools_params, save_warmup):
    dt = from_dict(
        {
            "posterior": data.obj,
            "posterior_predictive": data.obj,
            "sample_stats": data.obj,
            "prior": data.obj,
            "prior_predictive": data.obj,
            "sample_stats_prior": data.obj,
            "warmup_posterior": data.obj,
            "warmup_posterior_predictive": data.obj,
            "predictions": data.obj,
            "observed_data": eight_schools_params,
        },
        coords={
            "school": np.arange(8).tolist(),
        },
        pred_coords={
            "school_pred": list("abcdefgh"),
        },
        dims={"theta": ["school"], "eta": ["school"]},
        pred_dims={"theta": ["school_pred"], "eta": ["school_pred"]},
        save_warmup=save_warmup,
    )
    test_dict = {
        "posterior": [],
        "prior": [],
        "sample_stats": [],
        "posterior_predictive": [],
        "prior_predictive": [],
        "sample_stats_prior": [],
        "observed_data": ["J", "y", "sigma"],
        f"{'' if save_warmup else '~'}warmup_posterior": [],
        f"{'' if save_warmup else '~'}warmup_posterior_predictive": [],
    }
    fails = check_multiple_attrs(test_dict, dt)
    assert not fails
    check_var_names_coords_dims(dt.posterior)
    check_var_names_coords_dims(dt.posterior_predictive)
    check_var_names_coords_dims(dt.sample_stats)
    check_var_names_coords_dims(dt.prior)
    check_var_names_coords_dims(dt.prior_predictive)
    check_var_names_coords_dims(dt.sample_stats_prior)
    if save_warmup:
        check_var_names_coords_dims(dt.warmup_posterior)
        check_var_names_coords_dims(dt.warmup_posterior_predictive)

    pred_dims = dt.predictions.sizes["school_pred"]
    assert pred_dims == 8
    assert list(dt.predictions.school_pred.values) == list("abcdefgh")


def test_from_dict_auto_skip_event_dims():
    # create data
    rng = np.random.default_rng()
    data: dict[str, dict] = {
        "log_likelihood": {
            "y": rng.normal(size=(4, 100)),
        },
        "posterior_predictive": {
            "y": rng.normal(size=(4, 100, 8)),
        },
        "observed_data": {
            "y": rng.normal(size=8),
        },
    }

    dt = from_dict(data, dims={"y": ["school"]}, coords={"school": np.arange(8).tolist()})
    test_dict = {
        "log_likelihood": ["y"],
        "posterior_predictive": ["y"],
        "observed_data": ["y"],
    }
    fails = check_multiple_attrs(test_dict, dt)
    assert not fails
    assert "school" in dt.posterior_predictive.sizes
    assert "school" in dt.observed_data.sizes
    assert "school" not in dt.log_likelihood.sizes


def test_from_dict_attrs(data):
    dt = from_dict(
        {
            "posterior": data.obj,
            "sample_stats": data.obj,
        },
        name="Non centered eight",
        coords={
            "school": np.arange(8).tolist(),
        },
        dims={"theta": ["school"], "eta": ["school"]},
        attrs={"/": {"cool_atribute": "some metadata"}, "posterior": {"sampling_time": 20}},
    )
    test_dict: dict[str, list[str]] = {"posterior": [], "sample_stats": []}
    fails = check_multiple_attrs(test_dict, dt)
    assert not fails
    check_var_names_coords_dims(dt.posterior)
    check_var_names_coords_dims(dt.sample_stats)

    assert "sampling_time" in dt.posterior.attrs
    assert dt.posterior.attrs["sampling_time"] == 20
    assert "cool_atribute" in dt.attrs
    assert dt.attrs["cool_atribute"] == "some metadata"
    assert dt.name == "Non centered eight"
