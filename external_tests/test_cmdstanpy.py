# pylint: disable=redefined-outer-name, no-self-use
"""Integration tests for the cmdstanpy converter.

Whenever it is necessary to update the saved models for cmdstanpy,
delete the `external_tests/saved_models/cmdstanpy` folder and
run `pytest external_tests/test_cmdstanpy.py`.

All saved stan and csv files will be regenerated automatically.
"""

import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import pytest

from arviz_base import from_cmdstanpy
from arviz_base.testing import check_multiple_attrs

from .helpers import importorskip  # pylint: disable=unused-import


def _create_test_data(target_dir):
    """Create test data to local folder.

    This function is needed when test data needs to be updated.
    """
    import platform
    import shutil

    import cmdstanpy

    model_code = """
        data {
            int<lower=0> J;
            array[J] real y;
            array[J] real<lower=0> sigma;
        }

        parameters {
            real mu;
            real<lower=0> tau;
            array[2, J / 2] real eta;
        }

        transformed parameters {
            array[J] real theta;
            for (j in 1:J/2) {
                theta[j] = mu + tau * eta[1, j];
                theta[j + 4] = mu + tau * eta[2, j];
            }
        }

        model {
            mu ~ normal(0, 5);
            tau ~ cauchy(0, 5);
            eta[1] ~ normal(0, 1);
            eta[2] ~ normal(0, 1);
            y ~ normal(theta, sigma);
        }

        generated quantities {
            array[J] real log_lik;
            array[J] real y_hat;
            for (j in 1:J) {
                log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
                y_hat[j] = normal_rng(theta[j], sigma[j]);
            }
        }
    """
    stan_file = Path(target_dir) / "stan_model_test.stan"
    with open(stan_file, "w", encoding="utf8") as file_handle:
        file_handle.write(model_code)
    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    stan_data = {
        "J": 8,
        "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }
    fit_no_warmup = model.sample(
        data=stan_data, iter_sampling=100, iter_warmup=1000, save_warmup=False
    )
    fit_no_warmup.save_csvfiles(dir=target_dir)
    fit_files = {
        "cmdstanpy_eight_schools_nowarmup": [],
        "cmdstanpy_eight_schools_warmup": [],
    }
    for path_i in fit_no_warmup.runset.csv_files:
        path = Path(path_i)
        _, num = path.stem.rsplit("-", 1)
        new_path = path.parent / ("cmdstanpy_eight_schools_nowarmup-" + num + path.suffix)
        shutil.move(path, new_path)
        fit_files["cmdstanpy_eight_schools_nowarmup"].append(new_path)
    fit_warmup = model.sample(data=stan_data, iter_sampling=100, iter_warmup=500, save_warmup=True)
    fit_warmup.save_csvfiles(dir=target_dir)
    for path_i in fit_no_warmup.runset.csv_files:
        path = Path(path_i)
        _, num = path.stem.rsplit("-", 1)
        new_path = path.parent / ("cmdstanpy_eight_schools_warmup-" + num + path.suffix)
        shutil.move(path, new_path)
        fit_files["cmdstanpy_eight_schools_warmup"].append(new_path)
    stan_file.unlink()
    stan_file.with_suffix(".exe" if platform.system() == "Windows" else "").unlink()
    stan_file.with_suffix(".hpp").unlink(missing_ok=True)
    return fit_files


class TestDataCmdStanPy:
    @pytest.fixture(scope="session")
    @classmethod
    def data_directory(cls):
        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, "saved_models", "cmdstanpy")
        if not os.path.isdir(data_directory):
            os.makedirs(data_directory)
            _create_test_data(data_directory)
        return data_directory

    @pytest.fixture(scope="class")
    @classmethod
    def filepaths(cls, data_directory):
        files = {
            "nowarmup": glob(
                os.path.join(data_directory, "cmdstanpy_eight_schools_nowarmup-*_[1-4].csv")
            ),
            "warmup": glob(
                os.path.join(data_directory, "cmdstanpy_eight_schools_warmup-*_[1-4].csv")
            ),
        }
        return files

    @pytest.fixture(scope="class")
    @classmethod
    def data(cls, filepaths, tmp_path_factory):
        # Skip tests if cmdstanpy not installed
        cmdstanpy = importorskip("cmdstanpy")
        CmdStanModel = cmdstanpy.CmdStanModel  # pylint: disable=invalid-name
        CmdStanMCMC = cmdstanpy.CmdStanMCMC  # pylint: disable=invalid-name
        RunSet = cmdstanpy.stanfit.RunSet  # pylint: disable=invalid-name
        CmdStanArgs = cmdstanpy.model.CmdStanArgs  # pylint: disable=invalid-name
        SamplerArgs = cmdstanpy.model.SamplerArgs  # pylint: disable=invalid-name

        class Data:
            args = CmdStanArgs(
                "dummy.stan",
                "dummy.exe",
                list(range(1, 5)),
                method_args=SamplerArgs(iter_sampling=100),
            )
            runset_obj = RunSet(args)
            runset_obj._csv_files = filepaths["nowarmup"]  # pylint: disable=protected-access
            obj = CmdStanMCMC(runset_obj)
            obj._assemble_draws()  # pylint: disable=protected-access

            args_warmup = CmdStanArgs(
                "dummy.stan",
                "dummy.exe",
                list(range(1, 5)),
                method_args=SamplerArgs(iter_sampling=100, iter_warmup=500, save_warmup=True),
            )
            runset_obj_warmup = RunSet(args_warmup)
            runset_obj_warmup._csv_files = filepaths["warmup"]  # pylint: disable=protected-access
            obj_warmup = CmdStanMCMC(runset_obj_warmup)
            obj_warmup._assemble_draws()  # pylint: disable=protected-access

            temp_dir = tmp_path_factory.mktemp("stan_model")
            stan_file = temp_dir / "modified_dtype.stan"
            with open(stan_file, "w", encoding="utf8") as file_handle:
                modified_dtype_model = """
                generated quantities {
                    int eta;
                    array[8] int theta;
                }
                """
                file_handle.write(modified_dtype_model)
            exe_file = temp_dir / "modified_dtype"
            exe_file.touch()
            model = CmdStanModel(stan_file=stan_file, exe_file=exe_file, force_compile=False)

        return Data

    def get_inference_data(self, data, eight_schools_params):
        """vars as str."""
        return from_cmdstanpy(
            posterior=data.obj,
            posterior_predictive="y_hat",
            predictions="y_hat",
            prior=data.obj,
            prior_predictive="y_hat",
            observed_data={"y": eight_schools_params["y"]},
            constant_data={"y": eight_schools_params["y"]},
            predictions_constant_data={"y": eight_schools_params["y"]},
            log_likelihood={"y": "log_lik"},
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={
                "y": ["school"],
                "log_lik": ["school"],
                "y_hat": ["school"],
                "theta": ["school"],
            },
        )

    def get_inference_data2(self, data, eight_schools_params):
        """vars as lists."""
        return from_cmdstanpy(
            posterior=data.obj,
            posterior_predictive=["y_hat"],
            predictions=["y_hat", "log_lik"],
            prior=data.obj,
            prior_predictive=["y_hat"],
            observed_data={"y": eight_schools_params["y"]},
            constant_data=eight_schools_params,
            predictions_constant_data=eight_schools_params,
            log_likelihood=["log_lik", "y_hat"],
            coords={
                "school": np.arange(eight_schools_params["J"]),
                "log_lik_dim": np.arange(eight_schools_params["J"]),
            },
            dims={
                "eta": ["extra_dim", "half school"],
                "y": ["school"],
                "y_hat": ["school"],
                "theta": ["school"],
                "log_lik": ["log_lik_dim"],
            },
        )

    def get_inference_data3(self, data, eight_schools_params):
        """multiple vars as lists."""
        return from_cmdstanpy(
            posterior=data.obj,
            posterior_predictive=["y_hat", "log_lik"],
            prior=data.obj,
            prior_predictive=["y_hat", "log_lik"],
            observed_data={"y": eight_schools_params["y"]},
            coords={
                "school": np.arange(eight_schools_params["J"]),
                "half school": ["a", "b", "c", "d"],
                "extra_dim": ["x", "y"],
            },
            dims={
                "eta": ["extra_dim", "half school"],
                "y": ["school"],
                "y_hat": ["school"],
                "theta": ["school"],
                "log_lik": ["log_lik_dim"],
            },
            log_likelihood=True,
            dtypes=data.model,
        )

    def get_inference_data4(self, data, eight_schools_params):
        """multiple vars as lists."""
        return from_cmdstanpy(
            posterior=data.obj,
            posterior_predictive=None,
            prior=data.obj,
            prior_predictive=None,
            log_likelihood=True,
            observed_data={"y": eight_schools_params["y"]},
            coords=None,
            dims=None,
            dtypes={"eta": int, "theta": int},
        )

    def get_inference_data5(self, data, eight_schools_params):
        """multiple vars as lists."""
        return from_cmdstanpy(
            posterior=data.obj,
            posterior_predictive=None,
            prior=data.obj,
            prior_predictive=None,
            log_likelihood="log_lik",
            observed_data={"y": eight_schools_params["y"]},
            coords=None,
            dims=None,
            dtypes=data.model.code(),
        )

    def get_inference_data_warmup_true_is_true(self, data, eight_schools_params):
        """vars as str."""
        return from_cmdstanpy(
            posterior=data.obj_warmup,
            posterior_predictive="y_hat",
            predictions="y_hat",
            prior=data.obj_warmup,
            prior_predictive="y_hat",
            observed_data={"y": eight_schools_params["y"]},
            constant_data={"y": eight_schools_params["y"]},
            predictions_constant_data={"y": eight_schools_params["y"]},
            log_likelihood="log_lik",
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={
                "eta": ["extra_dim", "half school"],
                "y": ["school"],
                "log_lik": ["school"],
                "y_hat": ["school"],
                "theta": ["school"],
            },
            save_warmup=True,
        )

    def get_inference_data_warmup_false_is_true(self, data, eight_schools_params):
        """vars as str."""
        return from_cmdstanpy(
            posterior=data.obj,
            posterior_predictive="y_hat",
            predictions="y_hat",
            prior=data.obj,
            prior_predictive="y_hat",
            observed_data={"y": eight_schools_params["y"]},
            constant_data={"y": eight_schools_params["y"]},
            predictions_constant_data={"y": eight_schools_params["y"]},
            log_likelihood="log_lik",
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={
                "eta": ["extra_dim", "half school"],
                "y": ["school"],
                "log_lik": ["school"],
                "y_hat": ["school"],
                "theta": ["school"],
            },
            save_warmup=True,
        )

    def get_inference_data_warmup_true_is_false(self, data, eight_schools_params):
        """vars as str."""
        return from_cmdstanpy(
            posterior=data.obj_warmup,
            posterior_predictive="y_hat",
            predictions="y_hat",
            prior=data.obj_warmup,
            prior_predictive="y_hat",
            observed_data={"y": eight_schools_params["y"]},
            constant_data={"y": eight_schools_params["y"]},
            predictions_constant_data={"y": eight_schools_params["y"]},
            log_likelihood="log_lik",
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={
                "eta": ["extra_dim", "half school"],
                "y": ["school"],
                "log_lik": ["school"],
                "y_hat": ["school"],
                "theta": ["school"],
            },
            save_warmup=False,
        )

    def test_sampler_stats(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        test_dict = {"sample_stats": ["lp", "diverging"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        assert len(inference_data.sample_stats.lp.shape) == 2  # pylint: disable=no-member

    def test_inference_data(self, data, eight_schools_params):
        inference_data1 = self.get_inference_data(data, eight_schools_params)
        inference_data2 = self.get_inference_data2(data, eight_schools_params)
        inference_data3 = self.get_inference_data3(data, eight_schools_params)
        inference_data4 = self.get_inference_data4(data, eight_schools_params)
        inference_data5 = self.get_inference_data5(data, eight_schools_params)

        # inference_data 1
        test_dict = {
            "posterior": ["theta"],
            "predictions": ["y_hat"],
            "observed_data": ["y"],
            "constant_data": ["y"],
            "predictions_constant_data": ["y"],
            "log_likelihood": ["y", "~log_lik"],
            "prior": ["theta"],
        }
        fails = check_multiple_attrs(test_dict, inference_data1)
        assert not fails

        # inference_data 2
        test_dict = {
            "posterior_predictive": ["y_hat"],
            "predictions": ["y_hat", "log_lik"],
            "observed_data": ["y"],
            "sample_stats_prior": ["lp"],
            "sample_stats": ["lp"],
            "constant_data": list(eight_schools_params),
            "predictions_constant_data": list(eight_schools_params),
            "prior_predictive": ["y_hat"],
            "log_likelihood": ["log_lik", "y_hat"],
        }
        fails = check_multiple_attrs(test_dict, inference_data2)
        assert not fails

        # inference_data 3
        test_dict = {
            "posterior_predictive": ["y_hat"],
            "observed_data": ["y"],
            "sample_stats_prior": ["lp"],
            "sample_stats": ["lp"],
            "prior_predictive": ["y_hat"],
            "log_likelihood": ["log_lik"],
        }
        fails = check_multiple_attrs(test_dict, inference_data3)
        assert not fails
        assert inference_data3.posterior.eta.dtype.kind == "i"  # pylint: disable=no-member
        assert inference_data3.posterior.theta.dtype.kind == "i"  # pylint: disable=no-member

        # inference_data 4
        test_dict = {
            "posterior": ["eta", "mu", "theta"],
            "prior": ["theta"],
            "log_likelihood": ["log_lik"],
        }
        fails = check_multiple_attrs(test_dict, inference_data4)
        assert not fails
        assert len(inference_data4.posterior.theta.shape) == 3  # pylint: disable=no-member
        assert len(inference_data4.posterior.eta.shape) == 4  # pylint: disable=no-member
        assert len(inference_data4.posterior.mu.shape) == 2  # pylint: disable=no-member
        assert inference_data4.posterior.eta.dtype.kind == "i"  # pylint: disable=no-member
        assert inference_data4.posterior.theta.dtype.kind == "i"  # pylint: disable=no-member

        # inference_data 5
        test_dict = {
            "posterior": ["eta", "mu", "theta"],
            "prior": ["theta"],
            "log_likelihood": ["log_lik"],
        }
        fails = check_multiple_attrs(test_dict, inference_data5)
        assert inference_data5.posterior.eta.dtype.kind == "i"  # pylint: disable=no-member
        assert inference_data5.posterior.theta.dtype.kind == "i"  # pylint: disable=no-member

    def test_inference_data_warmup(self, data, eight_schools_params):
        inference_data_true_is_true = self.get_inference_data_warmup_true_is_true(
            data, eight_schools_params
        )
        inference_data_false_is_true = self.get_inference_data_warmup_false_is_true(
            data, eight_schools_params
        )
        inference_data_true_is_false = self.get_inference_data_warmup_true_is_false(
            data, eight_schools_params
        )
        inference_data_false_is_false = self.get_inference_data(data, eight_schools_params)
        # inference_data warmup
        test_dict = {
            "posterior": ["theta"],
            "predictions": ["y_hat"],
            "observed_data": ["y"],
            "constant_data": ["y"],
            "predictions_constant_data": ["y"],
            "log_likelihood": ["log_lik"],
            "prior": ["theta"],
            "warmup_posterior": ["theta"],
            "warmup_predictions": ["y_hat"],
            "warmup_log_likelihood": ["log_lik"],
            "warmup_prior": ["theta"],
        }
        fails = check_multiple_attrs(test_dict, inference_data_true_is_true)
        assert not fails
        # inference_data no warmup
        test_dict = {
            "posterior": ["theta"],
            "predictions": ["y_hat"],
            "observed_data": ["y"],
            "constant_data": ["y"],
            "predictions_constant_data": ["y"],
            "log_likelihood": ["log_lik"],
            "prior": ["theta"],
            "~warmup_posterior": [""],
            "~warmup_predictions": [""],
            "~warmup_log_likelihood": [""],
            "~warmup_prior": [""],
        }
        fails = check_multiple_attrs(test_dict, inference_data_false_is_true)
        assert not fails
        # inference_data no warmup
        test_dict = {
            "posterior": ["theta"],
            "predictions": ["y_hat"],
            "observed_data": ["y"],
            "constant_data": ["y"],
            "predictions_constant_data": ["y"],
            "log_likelihood": ["log_lik"],
            "prior": ["theta"],
            "~warmup_posterior": [""],
            "~warmup_predictions": [""],
            "~warmup_log_likelihood": [""],
            "~warmup_prior": [""],
        }
        fails = check_multiple_attrs(test_dict, inference_data_true_is_false)
        assert not fails
        # inference_data no warmup
        test_dict = {
            "posterior": ["theta"],
            "predictions": ["y_hat"],
            "observed_data": ["y"],
            "constant_data": ["y"],
            "predictions_constant_data": ["y"],
            "log_likelihood": ["y"],
            "prior": ["theta"],
            "~warmup_posterior": [""],
            "~warmup_predictions": [""],
            "~warmup_log_likelihood": [""],
            "~warmup_prior": [""],
        }
        fails = check_multiple_attrs(test_dict, inference_data_false_is_false)
        assert not fails


@pytest.fixture(scope="module")
def log_prior_fit(tmp_path_factory):
    """Read saved draws with explicit prior log densities through CmdStanPy's CSV parser."""
    cmdstanpy = importorskip("cmdstanpy")
    directory = tmp_path_factory.mktemp("log_prior")
    source = Path(__file__).parent / "saved_models" / "cmdstanpy"
    paths = []
    for path in sorted(source.glob("cmdstanpy_eight_schools_warmup-*.csv")):
        output = directory / path.name
        rows = []
        columns = None
        for line in path.read_text(encoding="utf8").splitlines():
            if line.startswith("#") or not line:
                rows.append(line)
            elif columns is None:
                columns = line.split(",")
                rows.append(
                    line
                    + ",log_prior_mu,log_prior_tau,"
                    + ",".join(f"log_prior_theta.{i}" for i in range(1, 9))
                )
            else:
                values = line.split(",")
                mu = float(values[columns.index("mu")])
                tau = float(values[columns.index("tau")])
                log_mu = -0.5 * (mu / 5) ** 2 - np.log(5 * np.sqrt(2 * np.pi))
                log_tau = np.log(2 / (5 * np.pi)) - np.log1p((tau / 5) ** 2)
                theta = np.array([float(values[columns.index(f"theta.{i}")]) for i in range(1, 9)])
                log_theta = -0.5 * ((theta - mu) / tau) ** 2 - np.log(tau * np.sqrt(2 * np.pi))
                rows.append(f"{line},{log_mu},{log_tau}," + ",".join(map(str, log_theta)))
        output.write_text("\n".join(rows) + "\n", encoding="utf8")
        paths.append(str(output))
    if hasattr(cmdstanpy, "from_output_files"):
        config = directory / "config.json"
        config.write_text(
            json.dumps(
                {
                    "model_name": "stan_model_test_model",
                    "stan_major_version": "2",
                    "stan_minor_version": "37",
                    "stan_patch_version": "0",
                    "method_config": {
                        "method": "sample",
                        "algorithm": "hmc",
                        "num_samples": 100,
                        "num_warmup": 500,
                        "save_warmup": True,
                        "max_depth": 10,
                    },
                }
            ),
            encoding="utf8",
        )
        return cmdstanpy.CmdStanMCMC.from_files(paths, config)
    return cmdstanpy.from_csv(paths)


@pytest.mark.parametrize(
    "selection",
    [
        "log_prior_mu",
        ["log_prior_mu", "log_prior_tau"],
        {"mu": "log_prior_mu", "tau": "log_prior_tau"},
    ],
)
@pytest.mark.parametrize("save_warmup", [False, True])
def test_log_prior(log_prior_fit, selection, save_warmup):
    """Selected densities preserve posterior sample ordering and optional warmup."""
    result = from_cmdstanpy(
        log_prior_fit,
        prior=log_prior_fit,
        log_prior=selection,
        log_likelihood={"y": "log_lik"},
        save_warmup=save_warmup,
        index_origin=1,
    )
    selected = {selection: selection} if isinstance(selection, str) else selection
    if isinstance(selected, list):
        selected = {name: name for name in selected}
    assert set(result.log_prior.data_vars) == set(selected)
    for name, original in selected.items():
        expected = log_prior_fit.stan_variable(original)
        np.testing.assert_allclose(result.log_prior[name].values.T.ravel(order="F"), expected)
        assert original not in result.posterior
        assert original in result.prior
        assert result.log_prior[name].dims == ("chain", "draw")
    assert result.log_prior.sizes == {"chain": 4, "draw": 100}
    np.testing.assert_array_equal(result.log_prior.draw, np.arange(1, 101))
    assert "y" in result.log_likelihood
    assert "mu" in result.posterior
    assert ("warmup_log_prior" in result) == save_warmup
    if save_warmup:
        assert result.warmup_log_prior.sizes == {"chain": 4, "draw": 500}
        for name, original in selected.items():
            expected = log_prior_fit.stan_variable(original, inc_warmup=True)
            warmup = expected.reshape((600, 4), order="F")[:500]
            np.testing.assert_allclose(result.warmup_log_prior[name].values.T, warmup)


def test_log_prior_default(log_prior_fit):
    """Prior densities stay in posterior unless explicitly selected."""
    result = from_cmdstanpy(log_prior_fit)
    assert "log_prior" not in result
    assert "log_prior_mu" in result.posterior


def test_log_prior_unknown_variable(log_prior_fit):
    """An invalid selection reports the missing Stan variable."""
    with pytest.raises(ValueError, match="unknown variable: missing_prior"):
        from_cmdstanpy(log_prior_fit, log_prior="missing_prior")


def test_log_prior_without_posterior(log_prior_fit):
    """Prior samples alone cannot provide prior density at posterior draws."""
    result = from_cmdstanpy(prior=log_prior_fit, log_prior="log_prior_mu")
    assert "log_prior" not in result
    assert "log_prior_mu" in result.prior


@pytest.mark.parametrize("save_warmup", [False, True])
def test_log_prior_vector_coordinates(log_prior_fit, save_warmup):
    """Vector parameter densities keep parameter dimensions and sample coordinates."""
    schools = [f"school_{i}" for i in range(8)]
    result = from_cmdstanpy(
        log_prior_fit,
        log_prior={"theta": "log_prior_theta"},
        log_likelihood={"y": "log_lik"},
        posterior_predictive=["y_hat"],
        coords={"school": schools},
        dims={"theta": ["school"], "y": ["school"]},
        save_warmup=save_warmup,
    )
    assert result.log_prior.theta.dims == ("chain", "draw", "school")
    np.testing.assert_array_equal(result.log_prior.school, schools)
    expected = log_prior_fit.stan_variable("log_prior_theta")
    actual = result.log_prior.theta.values.transpose(1, 0, 2).reshape((-1, 8), order="F")
    np.testing.assert_allclose(actual, expected)
    assert "theta" in result.posterior
    assert "log_prior_theta" not in result.posterior
    assert "y_hat" in result.posterior_predictive
    assert "y" in result.log_likelihood
    if save_warmup:
        expected = log_prior_fit.stan_variable("log_prior_theta", inc_warmup=True)
        expected = expected.reshape((600, 4, 8), order="F")[:500]
        np.testing.assert_allclose(
            result.warmup_log_prior.theta.values.transpose(1, 0, 2), expected
        )
        np.testing.assert_array_equal(result.warmup_log_prior.school, schools)
