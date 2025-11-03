# pylint: disable=no-member, invalid-name, redefined-outer-name
from collections import namedtuple

import numpy as np
import pytest

from arviz_base.io_numpyro import SVIWrapper, from_numpyro, from_numpyro_svi
from arviz_base.testing import check_multiple_attrs

from .helpers import importorskip, load_cached_models

# Skip all tests if jax or numpyro not installed
jax = importorskip("jax")
PRNGKey = jax.random.PRNGKey
numpyro = importorskip("numpyro")
Predictive = numpyro.infer.Predictive
autoguide = numpyro.infer.autoguide
numpyro.set_host_device_count(2)


class TestDataNumPyro:
    @pytest.fixture(scope="class", params=["numpyro", "numpyro_svi", "numpyro_svi_custom_guide"])
    def data(self, request, eight_schools_params, draws, chains):
        class Data:
            obj = load_cached_models(eight_schools_params, draws, chains, "numpyro")[request.param]

        return Data

    @pytest.fixture(scope="class")
    def predictions_params(self):
        """Predictions data for eight schools."""
        return {
            "J": 8,
            "sigma": np.array([5.0, 7.0, 12.0, 4.0, 6.0, 10.0, 3.0, 9.0]),
        }

    @pytest.fixture(scope="class")
    def predictions_data(self, data, predictions_params):
        """Generate predictions for predictions_params"""
        posterior = SVIWrapper(**data.obj) if isinstance(data.obj, dict) else data.obj
        posterior_samples = posterior.get_samples()
        model = posterior.sampler.model
        predictions = Predictive(model, posterior_samples)(
            PRNGKey(2), predictions_params["J"], predictions_params["sigma"]
        )
        return predictions

    def get_inference_data(
        self, data, eight_schools_params, predictions_data, predictions_params, infer_dims=False
    ):
        if isinstance(data.obj, dict):  # SVI cached data obj is a tuple
            posterior = SVIWrapper(**data.obj)
            from_numpyro_func = from_numpyro_svi
            posterior_kwarg = data.obj
        else:  # regular MCMC
            posterior = data.obj
            from_numpyro_func = from_numpyro
            posterior_kwarg = {"posterior": posterior}

        posterior_samples = posterior.get_samples()
        model = posterior.sampler.model
        posterior_predictive = Predictive(model, posterior_samples)(
            PRNGKey(1), eight_schools_params["J"], eight_schools_params["sigma"]
        )
        prior = Predictive(model, num_samples=500)(
            PRNGKey(2), eight_schools_params["J"], eight_schools_params["sigma"]
        )
        dims = {"theta": ["school"], "eta": ["school"], "obs": ["school"]}
        pred_dims = {"theta": ["school_pred"], "eta": ["school_pred"], "obs": ["school_pred"]}
        if infer_dims:
            dims = pred_dims = None

        predictions = predictions_data

        return from_numpyro_func(
            **posterior_kwarg,
            prior=prior,
            posterior_predictive=posterior_predictive,
            predictions=predictions,
            coords={
                "school": np.arange(eight_schools_params["J"]),
                "school_pred": np.arange(predictions_params["J"]),
            },
            dims=dims,
            pred_dims=pred_dims,
        )

    def test_inference_data_namedtuple(self, data):
        posterior = SVIWrapper(**data.obj) if isinstance(data.obj, dict) else data.obj
        samples = posterior.get_samples()
        Samples = namedtuple("Samples", samples)
        data_namedtuple = Samples(**samples)
        _old_fn = posterior.get_samples
        posterior.get_samples = lambda *args, **kwargs: data_namedtuple
        inference_data = from_numpyro(
            posterior=posterior,
            dims={},  # This mock test needs to turn off autodims like so or mock group_by_chain
        )
        assert isinstance(posterior.get_samples(), Samples)
        posterior.get_samples = _old_fn
        for key in samples:
            assert key in inference_data.posterior

    def test_inference_data(self, data, eight_schools_params, predictions_data, predictions_params):
        inference_data = self.get_inference_data(
            data, eight_schools_params, predictions_data, predictions_params
        )
        test_dict = {
            "posterior": ["mu", "tau", "eta"],
            "sample_stats": ["diverging"],
            "posterior_predictive": ["obs"],
            "predictions": ["obs"],
            "prior": ["mu", "tau", "eta"],
            "prior_predictive": ["obs"],
            "observed_data": ["obs"],
        }
        if isinstance(data.obj, dict):  # if its SVI, drop sample_stats check
            test_dict.pop("sample_stats")
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

        ## test dims
        dims = inference_data.posterior_predictive.sizes["school"]
        pred_dims = inference_data.predictions.sizes["school_pred"]
        assert dims == 8
        assert pred_dims == 8

    def test_inference_data_no_posterior(
        self, data, eight_schools_params, predictions_data, predictions_params
    ):
        posterior = SVIWrapper(**data.obj) if isinstance(data.obj, dict) else data.obj
        posterior_samples = posterior.get_samples()
        model = posterior.sampler.model
        posterior_predictive = Predictive(model, posterior_samples)(
            PRNGKey(1), eight_schools_params["J"], eight_schools_params["sigma"]
        )
        prior = Predictive(model, num_samples=500)(
            PRNGKey(2), eight_schools_params["J"], eight_schools_params["sigma"]
        )
        predictions = predictions_data
        constant_data = {"J": 8, "sigma": eight_schools_params["sigma"]}
        predictions_constant_data = predictions_params
        ##  only prior
        inference_data = from_numpyro(prior=prior)
        test_dict = {"prior": ["mu", "tau", "eta"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, f"only prior: {fails}"
        ## only posterior_predictive
        inference_data = from_numpyro(posterior_predictive=posterior_predictive)
        test_dict = {"posterior_predictive": ["obs"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, f"only posterior_predictive: {fails}"
        ## only predictions
        inference_data = from_numpyro(predictions=predictions)
        test_dict = {"predictions": ["obs"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, f"only predictions: {fails}"
        ## only constant_data
        inference_data = from_numpyro(constant_data=constant_data)
        test_dict = {"constant_data": ["J", "sigma"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, f"only constant_data: {fails}"
        ## only predictions_constant_data
        inference_data = from_numpyro(predictions_constant_data=predictions_constant_data)
        test_dict = {"predictions_constant_data": ["J", "sigma"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, f"only predictions_constant_data: {fails}"
        prior and posterior_predictive
        idata = from_numpyro(
            prior=prior,
            posterior_predictive=posterior_predictive,
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={"theta": ["school"], "eta": ["school"]},
        )
        test_dict = {"posterior_predictive": ["obs"], "prior": ["mu", "tau", "eta", "obs"]}
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails, f"prior and posterior_predictive: {fails}"

    def test_inference_data_only_posterior(self, data):
        kwargs = data.obj if isinstance(data.obj, dict) else {"posterior": data.obj}
        from_numpyro_func = from_numpyro_svi if isinstance(data.obj, dict) else from_numpyro
        idata = from_numpyro_func(**kwargs)
        test_dict = {
            "posterior": ["mu", "tau", "eta"],
            "sample_stats": ["diverging"],
        }
        if isinstance(data.obj, dict):
            test_dict.pop("sample_stats")
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails

    def test_multiple_observed_rv(self):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        rng = np.random.default_rng()
        y1 = rng.normal(size=10)
        y2 = rng.normal(size=100)

        def model_example_multiple_obs(y1=None, y2=None):
            x = numpyro.sample("x", dist.Normal(1, 3))
            numpyro.sample("y1", dist.Normal(x, 1), obs=y1)
            numpyro.sample("y2", dist.Normal(x, 1), obs=y2)

        nuts_kernel = NUTS(model_example_multiple_obs)
        mcmc = MCMC(nuts_kernel, num_samples=10, num_warmup=2)
        mcmc.run(PRNGKey(0), y1=y1, y2=y2)
        inference_data = from_numpyro(mcmc)
        test_dict = {
            "posterior": ["x"],
            "sample_stats": ["diverging"],
            "observed_data": ["y1", "y2"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        assert not hasattr(inference_data.sample_stats, "log_likelihood")

    def test_inference_data_constant_data(self):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        x1 = 10
        x2 = 12
        rng = np.random.default_rng()
        y1 = rng.normal(size=10)

        def model_constant_data(x, y1=None):
            _x = numpyro.sample("x", dist.Normal(1, 3))
            numpyro.sample("y1", dist.Normal(x * _x, 1), obs=y1)

        nuts_kernel = NUTS(model_constant_data)
        mcmc = MCMC(nuts_kernel, num_samples=10, num_warmup=2)
        mcmc.run(PRNGKey(0), x=x1, y1=y1)
        posterior = mcmc.get_samples()
        posterior_predictive = Predictive(model_constant_data, posterior)(PRNGKey(1), x1)
        predictions = Predictive(model_constant_data, posterior)(PRNGKey(2), x2)
        inference_data = from_numpyro(
            mcmc,
            posterior_predictive=posterior_predictive,
            predictions=predictions,
            constant_data={"x1": x1},
            predictions_constant_data={"x2": x2},
        )
        test_dict = {
            "posterior": ["x"],
            "posterior_predictive": ["y1"],
            "sample_stats": ["diverging"],
            "predictions": ["y1"],
            "observed_data": ["y1"],
            "constant_data": ["x1"],
            "predictions_constant_data": ["x2"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    def test_inference_data_num_chains(self, predictions_data, chains):
        predictions = predictions_data
        inference_data = from_numpyro(predictions=predictions, num_chains=chains)
        nchains = inference_data.predictions.sizes["chain"]
        assert nchains == chains

    @pytest.mark.parametrize("nchains", [1, 2])
    @pytest.mark.parametrize("thin", [1, 2, 3, 5, 10])
    def test_mcmc_with_thinning(self, nchains, thin):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        rng = np.random.default_rng()
        x = rng.normal(10, 3, size=100)

        def model(x):
            numpyro.sample(
                "x",
                dist.Normal(
                    numpyro.sample("loc", dist.Uniform(0, 20)),
                    numpyro.sample("scale", dist.Uniform(0, 20)),
                ),
                obs=x,
            )

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=400, num_chains=nchains, thinning=thin)
        mcmc.run(PRNGKey(0), x=x)

        inference_data = from_numpyro(mcmc)
        assert inference_data.posterior["loc"].shape == (nchains, 400 // thin)

    def test_mcmc_improper_uniform(self):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        def model():
            x = numpyro.sample("x", dist.ImproperUniform(dist.constraints.positive, (), ()))
            return numpyro.sample("y", dist.Normal(x, 1), obs=1.0)

        mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=10)
        mcmc.run(PRNGKey(0))
        inference_data = from_numpyro(mcmc)
        assert inference_data.observed_data

    @pytest.mark.parametrize(
        "svi,guide_fn",
        [
            (False, None),  # MCMC, guide ignored
            (True, autoguide.AutoDelta),  # SVI with AutoDelta
            (True, autoguide.AutoNormal),  # SVI with AutoNormal
            (True, "custom"),  # SVI with custom guide
        ],
    )
    def test_infer_dims(self, svi, guide_fn):
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        def model():
            # note: group2 gets assigned dim=-1 and group1 is assigned dim=-2
            with numpyro.plate("group2", 5), numpyro.plate("group1", 10):
                _ = numpyro.sample("param", dist.Normal(0, 1))

        def guide():
            loc = numpyro.param("param_loc", jnp.zeros((10, 5)))
            scale = numpyro.param(
                "param_scale", jnp.ones((10, 5)), constraint=dist.constraints.positive
            )
            with numpyro.plate("group2", 5), numpyro.plate("group1", 10):
                numpyro.sample("param", dist.Normal(loc, scale))

        if guide_fn == "custom":
            guide_fn = guide

        result = self._run_inference(model, svi=svi, guide_fn=guide_fn)
        from_numpyro_func = from_numpyro_svi if svi else from_numpyro
        sample_dims = ("sample",) if svi else ("chain", "draw")

        inference_data = from_numpyro_func(
            **result, coords={"group1": np.arange(10), "group2": np.arange(5)}
        )
        assert inference_data.posterior.param.dims == sample_dims + ("group1", "group2")
        assert all(dim in inference_data.posterior.param.coords for dim in ("group1", "group2"))

    @pytest.mark.parametrize(
        "svi,guide_fn",
        [
            (False, None),  # MCMC, guide ignored
            (True, autoguide.AutoDelta),  # SVI with AutoDelta
            (True, autoguide.AutoNormal),  # SVI with AutoNormal
            (True, "custom"),  # SVI with custom guide
        ],
    )
    def test_infer_unsorted_dims(self, svi, guide_fn):
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        def model():
            group1_plate = numpyro.plate("group1", 10, dim=-1)
            group2_plate = numpyro.plate("group2", 5, dim=-2)

            # the plate contexts are entered in a different order than the pre-defined dims
            # we should make sure this still works because the trace has all of the info it needs
            with group2_plate, group1_plate:
                _ = numpyro.sample("param", dist.Normal(0, 1))

        def guide():
            loc = numpyro.param("param_loc", jnp.zeros((5, 10)))
            scale = numpyro.param(
                "param_scale", jnp.ones((5, 10)), constraint=dist.constraints.positive
            )
            group1_plate = numpyro.plate("group1", 10, dim=-1)
            group2_plate = numpyro.plate("group2", 5, dim=-2)
            with group2_plate, group1_plate:
                numpyro.sample("param", dist.Normal(loc, scale))

        if guide_fn == "custom":
            guide_fn = guide

        result = self._run_inference(model, svi=svi, guide_fn=guide_fn)
        from_numpyro_func = from_numpyro_svi if svi else from_numpyro
        sample_dims = ("sample",) if svi else ("chain", "draw")

        inference_data = from_numpyro_func(
            **result, coords={"group1": np.arange(10), "group2": np.arange(5)}
        )
        assert inference_data.posterior.param.dims == sample_dims + ("group2", "group1")
        assert all(dim in inference_data.posterior.param.coords for dim in ("group1", "group2"))

    @pytest.mark.parametrize(
        "svi,guide_fn",
        [
            (False, None),  # MCMC, guide ignored
            (True, autoguide.AutoDelta),  # SVI with AutoDelta
            (True, autoguide.AutoNormal),  # SVI with AutoNormal
            (True, "custom"),  # SVI with custom guide
        ],
    )
    def test_infer_dims_no_coords(self, svi, guide_fn):
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        def model():
            with numpyro.plate("group", 5):
                _ = numpyro.sample("param", dist.Normal(0, 1))

        def guide():
            loc = numpyro.param("param_loc", jnp.zeros(5))
            scale = numpyro.param("param_scale", jnp.ones(5), constraint=dist.constraints.positive)
            with numpyro.plate("group", 5):
                numpyro.sample("param", dist.Normal(loc, scale))

        if guide_fn == "custom":
            guide_fn = guide

        result = self._run_inference(model, svi=svi, guide_fn=guide_fn)
        from_numpyro_func = from_numpyro_svi if svi else from_numpyro
        sample_dims = ("sample",) if svi else ("chain", "draw")

        inference_data = from_numpyro_func(**result)
        assert inference_data.posterior.param.dims == sample_dims + ("group",)

    @pytest.mark.parametrize(
        "svi,guide_fn",
        [
            (False, None),  # MCMC, guide ignored
            (True, autoguide.AutoDelta),  # SVI with AutoDelta
            (True, autoguide.AutoNormal),  # SVI with AutoNormal
            (True, "custom"),  # SVI with custom guide
        ],
    )
    def test_event_dims(self, svi, guide_fn):
        import numpyro
        import numpyro.distributions as dist

        def model():
            _ = numpyro.sample(
                "gamma", dist.ZeroSumNormal(1, event_shape=(10,)), infer={"event_dims": ["groups"]}
            )

        def guide():
            scale = numpyro.param(
                "gamma_scale",
                1.0,
                constraint=dist.constraints.positive,
            )
            numpyro.sample("gamma", dist.ZeroSumNormal(scale, event_shape=(10,)))

        if guide_fn == "custom":
            guide_fn = guide

        result = self._run_inference(model, svi=svi, guide_fn=guide_fn)
        from_numpyro_func = from_numpyro_svi if svi else from_numpyro
        sample_dims = ("sample",) if svi else ("chain", "draw")

        inference_data = from_numpyro_func(**result, coords={"groups": np.arange(10)})
        assert inference_data.posterior.gamma.dims == sample_dims + ("groups",)
        assert "groups" in inference_data.posterior.gamma.coords

    @pytest.mark.parametrize(
        "svi,guide_fn",
        [
            (False, None),  # MCMC, guide ignored
            (True, autoguide.AutoDelta),  # SVI with AutoDelta
            (True, autoguide.AutoNormal),  # SVI with AutoNormal
            (True, "custom"),  # SVI with custom guide
        ],
    )
    def test_inferred_dims_univariate(self, svi, guide_fn):
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        def model():
            alpha = numpyro.sample("alpha", dist.Normal(0, 1))
            sigma = numpyro.sample("sigma", dist.HalfNormal(1))
            with numpyro.plate("obs_idx", 3):
                # mu is plated by obs_idx, but isnt broadcasted to the plate shape
                # the expected behavior is that this should cause a failure
                mu = numpyro.deterministic("mu", alpha)
                return numpyro.sample("y", dist.Normal(mu, sigma), obs=jnp.array([-1, 0, 1]))

        def guide():
            alpha_loc = numpyro.param("alpha_loc", jnp.array(0.0))
            alpha_scale = numpyro.param(
                "alpha_scale", jnp.array(1.0), constraint=dist.constraints.positive
            )
            sigma_loc = numpyro.param(
                "sigma_loc", jnp.array(1.0), constraint=dist.constraints.positive
            )

            alpha = numpyro.sample("alpha", dist.Normal(alpha_loc, alpha_scale))
            numpyro.sample("sigma", dist.HalfNormal(sigma_loc))
            with numpyro.plate("obs_idx", 3):
                numpyro.deterministic("mu", alpha)

        if guide_fn == "custom":
            guide_fn = guide

        result = self._run_inference(model, svi=svi, guide_fn=guide_fn)
        from_numpyro_func = from_numpyro_svi if svi else from_numpyro
        with pytest.raises(ValueError):
            from_numpyro_func(**result, coords={"obs_idx": np.arange(3)})

    @pytest.mark.parametrize(
        "svi,guide_fn",
        [
            (False, None),  # MCMC, guide ignored
            (True, autoguide.AutoDelta),  # SVI with AutoDelta
            (True, autoguide.AutoNormal),  # SVI with AutoNormal
            (True, "custom"),  # SVI with custom guide
        ],
    )
    def test_extra_event_dims(self, svi, guide_fn):
        import numpyro
        import numpyro.distributions as dist

        def model():
            gamma = numpyro.sample("gamma", dist.ZeroSumNormal(1, event_shape=(10,)))
            _ = numpyro.deterministic("gamma_plus1", gamma + 1)

        def guide():
            scale = numpyro.param(
                "gamma_scale",
                1.0,
                constraint=dist.constraints.positive,
            )
            gamma = numpyro.sample("gamma", dist.ZeroSumNormal(scale, event_shape=(10,)))
            numpyro.deterministic("gamma_plus1", gamma + 1)

        if guide_fn == "custom":
            guide_fn = guide

        result = self._run_inference(model, svi=svi, guide_fn=guide_fn)
        from_numpyro_func = from_numpyro_svi if svi else from_numpyro
        sample_dims = ("sample",) if svi else ("chain", "draw")
        inference_data = from_numpyro_func(
            **result, coords={"groups": np.arange(10)}, extra_event_dims={"gamma_plus1": ["groups"]}
        )
        assert inference_data.posterior.gamma_plus1.dims == sample_dims + ("groups",)
        assert "groups" in inference_data.posterior.gamma_plus1.coords

    def test_predictions_infer_dims(
        self, data, eight_schools_params, predictions_data, predictions_params
    ):
        inference_data = self.get_inference_data(
            data, eight_schools_params, predictions_data, predictions_params, infer_dims=True
        )
        sample_dims = ("sample",) if isinstance(data.obj, dict) else ("chain", "draw")
        assert inference_data.predictions.obs.dims == (sample_dims + ("J",))
        assert "J" in inference_data.predictions.obs.coords

    def _run_inference(self, model, svi, guide_fn):
        from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
        from numpyro.optim import Adam

        if svi:
            is_autoguide = isinstance(guide_fn, type) and issubclass(guide_fn, autoguide.AutoGuide)
            guide = guide_fn(model) if is_autoguide else guide_fn
            svi = SVI(model, guide=guide, optim=Adam(0.05), loss=Trace_ELBO())
            svi_result = svi.run(PRNGKey(0), 10)
            return {
                "svi": svi,
                "svi_result": svi_result,
                "model": None if is_autoguide else model,
            }

        else:
            mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=10)
            mcmc.run(PRNGKey(0))
            return {"posterior": mcmc}


class TestSVIWrapper:
    @pytest.fixture(scope="class", params=["numpyro_svi", "numpyro_svi_custom_guide"])
    def data(self, request, eight_schools_params, draws, chains):
        class Data:
            obj = load_cached_models(eight_schools_params, draws, chains, "numpyro")[request.param]

        return Data

    def test_init_without_args_kwargs(self):
        from numpyro.infer import Trace_ELBO
        from numpyro.infer.svi import SVI, SVIRunResult
        from numpyro.optim import Adam

        model = guide = lambda x: x
        svi = SVI(model, guide, optim=Adam(0.05), loss=Trace_ELBO())
        svi_result = SVIRunResult(params=jax.numpy.ones(5), state=None, losses=jax.numpy.zeros(10))

        posterior = SVIWrapper(svi, svi_result=svi_result)
        assert isinstance(posterior._args, tuple)
        assert isinstance(posterior._kwargs, dict)

    def test_get_samples(self, data, eight_schools_params):
        svi_posterior = SVIWrapper(
            data.obj["svi"], svi_result=data.obj["svi_result"], model_kwargs=eight_schools_params
        )
        out = svi_posterior.get_samples(seed=0)
        assert isinstance(out, dict)
        for v in out.values():  # values are array-like
            assert isinstance(v, (jax.numpy.ndarray | np.ndarray))

    def test_sampler_attr(self, data, eight_schools_params):
        svi_posterior = SVIWrapper(
            data.obj["svi"], svi_result=data.obj["svi_result"], model_kwargs=eight_schools_params
        )
        assert hasattr(svi_posterior, "sampler")
        assert hasattr(svi_posterior.sampler, "model")
