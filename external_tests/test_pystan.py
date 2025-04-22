# pylint: disable=no-member, invalid-name, redefined-outer-name, too-many-function-args
import importlib
import os

import numpy as np
import pytest

from arviz_base import from_pystan
from arviz_base.io_pystan import get_draws_stan3
from arviz_base.testing import check_multiple_attrs

from .helpers import (  # pylint: disable=unused-import
    # chains,
    # check_multiple_attrs,
    # draws,
    # eight_schools_params,
    load_cached_models,
    # pystan_version,
)

# Check if either pystan or pystan3 is installed
pystan_installed = (importlib.util.find_spec("pystan") is not None) or (
    importlib.util.find_spec("stan") is not None
)


@pytest.mark.skipif(
    not (pystan_installed or "ARVIZ_REQUIRE_ALL_DEPS" in os.environ),
    reason="test requires pystan/pystan3 which is not installed",
)
class TestDataPyStan:
    @pytest.fixture(scope="class")
    def data(self, eight_schools_params, draws, chains):
        class Data:
            model, obj = load_cached_models(eight_schools_params, draws, chains, "pystan")["pystan"]

        return Data

    def get_inference_data(self, data, eight_schools_params):
        """vars as str."""
        return from_pystan(
            posterior=data.obj,
            posterior_predictive="y_hat",
            predictions="y_hat",  # wrong, but fine for testing
            prior=data.obj,
            prior_predictive="y_hat",
            observed_data="y",
            constant_data="sigma",
            predictions_constant_data="sigma",  # wrong, but fine for testing
            log_likelihood={"y": "log_lik"},
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={
                "theta": ["school"],
                "y": ["school"],
                "sigma": ["school"],
                "y_hat": ["school"],
                "eta": ["school"],
            },
            posterior_model=data.model,
            prior_model=data.model,
        )

    def get_inference_data2(self, data, eight_schools_params):
        """vars as lists."""
        return from_pystan(
            posterior=data.obj,
            posterior_predictive=["y_hat"],
            predictions=["y_hat"],  # wrong, but fine for testing
            prior=data.obj,
            prior_predictive=["y_hat"],
            observed_data=["y"],
            log_likelihood="log_lik",
            coords={
                "school": np.arange(eight_schools_params["J"]),
                "log_likelihood_dim": np.arange(eight_schools_params["J"]),
            },
            dims={
                "theta": ["school"],
                "y": ["school"],
                "y_hat": ["school"],
                "eta": ["school"],
                "log_lik": ["log_likelihood_dim"],
            },
            posterior_model=data.model,
            prior_model=data.model,
        )

    def get_inference_data3(self, data, eight_schools_params):
        """multiple vars as lists."""
        return from_pystan(
            posterior=data.obj,
            posterior_predictive=["y_hat", "log_lik"],  # wrong, but fine for testing
            predictions=["y_hat", "log_lik"],  # wrong, but fine for testing
            prior=data.obj,
            prior_predictive=["y_hat", "log_lik"],  # wrong, but fine for testing
            constant_data=["sigma", "y"],  # wrong, but fine for testing
            predictions_constant_data=["sigma", "y"],  # wrong, but fine for testing
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={
                "theta": ["school"],
                "y": ["school"],
                "sigma": ["school"],
                "y_hat": ["school"],
                "eta": ["school"],
            },
            posterior_model=data.model,
            prior_model=data.model,
        )

    def get_inference_data4(self, data):
        """minimal input."""
        return from_pystan(
            posterior=data.obj,
            posterior_predictive=None,
            prior=data.obj,
            prior_predictive=None,
            coords=None,
            dims=None,
            posterior_model=data.model,
            log_likelihood=[],
            prior_model=data.model,
            save_warmup=True,
        )

    def get_inference_data5(self, data):
        """minimal input."""
        return from_pystan(
            posterior=data.obj,
            posterior_predictive=None,
            prior=data.obj,
            prior_predictive=None,
            coords=None,
            dims=None,
            posterior_model=data.model,
            log_likelihood=False,
            prior_model=data.model,
            save_warmup=True,
            dtypes={"eta": int},
        )

    def test_sampler_stats(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        test_dict = {"sample_stats": ["diverging"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    def test_inference_data(self, data, eight_schools_params):
        inference_data1 = self.get_inference_data(data, eight_schools_params)
        inference_data2 = self.get_inference_data2(data, eight_schools_params)
        inference_data3 = self.get_inference_data3(data, eight_schools_params)
        inference_data4 = self.get_inference_data4(data)
        inference_data5 = self.get_inference_data5(data)
        ## inference_data 1
        test_dict = {
            "posterior": ["theta", "~log_lik"],
            "posterior_predictive": ["y_hat"],
            "predictions": ["y_hat"],
            "observed_data": ["y"],
            "constant_data": ["sigma"],
            "predictions_constant_data": ["sigma"],
            "sample_stats": ["diverging", "lp"],
            "log_likelihood": ["y", "~log_lik"],
            "prior": ["theta"],
        }
        fails = check_multiple_attrs(test_dict, inference_data1)
        assert not fails
        ## inference_data 2
        test_dict = {
            "posterior_predictive": ["y_hat"],
            "predictions": ["y_hat"],
            "observed_data": ["y"],
            "sample_stats_prior": ["diverging"],
            "sample_stats": ["diverging", "lp"],
            # "log_likelihood": ["log_lik"],
            "prior_predictive": ["y_hat"],
        }
        fails = check_multiple_attrs(test_dict, inference_data2)
        assert not fails
        assert any(
            item in inference_data2.posterior.attrs for item in ["stan_code", "program_code"]
        )
        assert any(
            item in inference_data2.sample_stats.attrs for item in ["stan_code", "program_code"]
        )
        ## inference_data 3
        test_dict = {
            "posterior_predictive": ["y_hat", "log_lik"],
            "predictions": ["y_hat", "log_lik"],
            "constant_data": ["sigma", "y"],
            "predictions_constant_data": ["sigma", "y"],
            "sample_stats_prior": ["diverging"],
            "sample_stats": ["diverging", "lp"],
            # "log_likelihood": ["log_lik"],
            "prior_predictive": ["y_hat", "log_lik"],
        }
        fails = check_multiple_attrs(test_dict, inference_data3)
        assert not fails
        # inference_data 4
        test_dict = {
            "posterior": ["theta"],
            "prior": ["theta"],
            "sample_stats": ["diverging", "lp"],
            # "~log_likelihood": [""],
            "warmup_posterior": ["theta"],
            "warmup_sample_stats": ["diverging", "lp"],
        }
        fails = check_multiple_attrs(test_dict, inference_data4)
        assert not fails
        # inference_data 5
        test_dict = {
            "posterior": ["theta"],
            "prior": ["theta"],
            "sample_stats": ["diverging", "lp"],
            "~log_likelihood": [""],
            "warmup_posterior": ["theta"],
            "warmup_sample_stats": ["diverging", "lp"],
        }
        fails = check_multiple_attrs(test_dict, inference_data5)
        assert not fails
        assert inference_data5.posterior.eta.dtype.kind == "i"

    def test_empty_parameter(self):
        model_code = """
            parameters {
                real y;
                vector[3] x;
                vector[0] a;
                vector[2] z;
            }
            model {
                y ~ normal(0,1);
            }
        """
        import stan  # pylint: disable=import-error

        model = stan.build(model_code)
        fit = model.sample(num_samples=500, num_chains=2)

        posterior = from_pystan(posterior=fit)
        test_dict = {"posterior": ["y", "x", "z", "~a"], "sample_stats": ["diverging"]}
        fails = check_multiple_attrs(test_dict, posterior)
        assert not fails

    def test_get_draws(self, data):
        fit = data.obj
        draws, _ = get_draws_stan3(fit, variables=["theta", "theta"])
        assert draws.get("theta") is not None
