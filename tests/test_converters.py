# pylint: disable=no-member, no-self-use, invalid-name, redefined-outer-name
import os

import numpy as np
import pytest
import xarray as xr

from arviz_base import convert_to_dataset, convert_to_datatree, extract

netcdf_nightlies_skip = pytest.mark.skipif(
    os.environ.get("NIGHTLIES", False) == "TRUE",
    reason="Skip netcdf4 dependent tests from nightlies as it generally takes longer to update.",
)


def test_1d_dataset():
    size = 100
    rng = np.random.default_rng()
    dataset = convert_to_dataset(rng.normal(size=size), sample_dims=["sample"])
    assert len(dataset.data_vars) == 1

    assert set(dataset.coords) == {"sample"}
    assert dataset.sizes["sample"] == size


def test_warns_bad_shape():
    rng = np.random.default_rng()
    ary = rng.normal(size=(100, 4))
    # Shape should be (chain, draw, *shape)
    with pytest.warns(UserWarning, match="Found chain dimension to be longer than draw"):
        convert_to_dataset(ary, sample_dims=("chain", "draw"))
    # Shape should now be (draw, chain, *shape)
    dataset = convert_to_dataset(ary, sample_dims=("draw", "chain"))
    assert dataset.sizes["chain"] == 4
    assert dataset.sizes["draw"] == 100


def test_nd_to_dataset():
    shape = (1, 20, 3, 4, 5)
    rng = np.random.default_rng()
    dataset = convert_to_dataset(rng.normal(size=shape), sample_dims=("chain", "draw", "pred_id"))
    assert len(dataset.data_vars) == 1
    var_name = list(dataset.data_vars)[0]

    assert len(dataset.coords) == len(shape)
    assert dataset.sizes["chain"] == shape[0]
    assert dataset.sizes["draw"] == shape[1]
    assert dataset.sizes["pred_id"] == shape[2]
    assert dataset[var_name].shape == shape


def test_nd_to_datatree():
    shape = (1, 2, 3, 4, 5)
    rng = np.random.default_rng()
    data = convert_to_datatree(rng.normal(size=shape), group="prior")
    assert "/prior" in data.groups
    prior = data["prior"]
    assert len(prior.data_vars) == 1
    var_name = list(prior.data_vars)[0]

    assert len(prior.coords) == len(shape)
    assert prior.sizes["chain"] == shape[0]
    assert prior.sizes["draw"] == shape[1]
    assert prior[var_name].shape == shape


def test_more_chains_than_draws():
    shape = (10, 4)
    rng = np.random.default_rng()
    with pytest.warns(UserWarning):
        data = convert_to_datatree(rng.normal(size=shape), group="prior")
    assert "/prior" in data.groups
    prior = data["prior"]
    assert len(prior.data_vars) == 1
    var_name = list(prior.data_vars)[0]

    assert len(prior.coords) == len(shape)
    assert prior.sizes["chain"] == shape[0]
    assert prior.sizes["draw"] == shape[1]
    assert prior[var_name].shape == shape


class TestConvertToDataset:
    @pytest.fixture(scope="class")
    def data(self):
        rng = np.random.default_rng()

        # pylint: disable=attribute-defined-outside-init
        class Data:
            datadict = {
                "a": rng.normal(size=(1, 100)),
                "b": rng.normal(size=(1, 100, 10)),
                "c": rng.normal(size=(1, 100, 3, 4)),
            }
            coords = {"c1": np.arange(3), "c2": np.arange(4), "b1": np.arange(10)}
            dims = {"b": ["b1"], "c": ["c1", "c2"]}

        return Data

    def test_use_all(self, data):
        dataset = convert_to_dataset(data.datadict, coords=data.coords, dims=data.dims)
        assert set(dataset.data_vars) == {"a", "b", "c"}
        assert set(dataset.coords) == {"chain", "draw", "c1", "c2", "b1"}

        assert set(dataset.a.coords) == {"chain", "draw"}
        assert set(dataset.b.coords) == {"chain", "draw", "b1"}
        assert set(dataset.c.coords) == {"chain", "draw", "c1", "c2"}

    def test_missing_coords(self, data):
        dataset = convert_to_dataset(data.datadict, coords=None, dims=data.dims)
        assert set(dataset.data_vars) == {"a", "b", "c"}
        assert set(dataset.coords) == {"chain", "draw", "c1", "c2", "b1"}

        assert set(dataset.a.coords) == {"chain", "draw"}
        assert set(dataset.b.coords) == {"chain", "draw", "b1"}
        assert set(dataset.c.coords) == {"chain", "draw", "c1", "c2"}

    def test_missing_dims(self, data):
        # missing dims
        coords = {"c_dim_0": np.arange(3), "c_dim_1": np.arange(4), "b_dim_0": np.arange(10)}
        dataset = convert_to_dataset(data.datadict, coords=coords, dims=None)
        assert set(dataset.data_vars) == {"a", "b", "c"}
        assert set(dataset.coords) == {"chain", "draw", "c_dim_0", "c_dim_1", "b_dim_0"}

        assert set(dataset.a.coords) == {"chain", "draw"}
        assert set(dataset.b.coords) == {"chain", "draw", "b_dim_0"}
        assert set(dataset.c.coords) == {"chain", "draw", "c_dim_0", "c_dim_1"}

    def test_skip_dim_0(self, data):
        dims = {"c": [None, "c2"]}
        coords = {"c_dim_0": np.arange(3), "c2": np.arange(4), "b_dim_0": np.arange(10)}
        dataset = convert_to_dataset(data.datadict, coords=coords, dims=dims)
        assert set(dataset.data_vars) == {"a", "b", "c"}
        assert set(dataset.coords) == {"chain", "draw", "c_dim_0", "c2", "b_dim_0"}

        assert set(dataset.a.coords) == {"chain", "draw"}
        assert set(dataset.b.coords) == {"chain", "draw", "b_dim_0"}
        assert set(dataset.c.coords) == {"chain", "draw", "c_dim_0", "c2"}


def test_convert_to_dataset_idempotent():
    rng = np.random.default_rng()
    first = convert_to_dataset(rng.normal(size=(1, 100)))
    second = convert_to_dataset(first)
    assert first.equals(second)


def test_convert_to_datatree_idempotent():
    rng = np.random.default_rng()
    first = convert_to_datatree(rng.normal(size=(1, 100)), group="prior")
    second = convert_to_datatree(first)
    assert first.prior is second.prior


@netcdf_nightlies_skip
def test_convert_to_datatree_from_file(tmpdir):
    rng = np.random.default_rng()
    first = convert_to_datatree(rng.normal(size=(1, 100)), group="prior")
    filename = str(tmpdir.join("test_file.nc"))
    first.to_netcdf(filename)
    second = convert_to_datatree(filename)
    assert first.prior.equals(second.prior)


def test_convert_to_datatree_bad():
    with pytest.raises(ValueError):
        convert_to_datatree(1)


@netcdf_nightlies_skip
def test_convert_to_dataset_bad(tmpdir):
    rng = np.random.default_rng()
    first = convert_to_datatree(rng.normal(size=(1, 100)), group="prior")
    filename = str(tmpdir.join("test_file.nc"))
    first.to_netcdf(filename)
    with pytest.raises(ValueError):
        convert_to_dataset(filename, group="bar")


class TestDataConvert:
    @pytest.fixture(scope="class")
    def data(self, draws, chains):
        rng = np.random.default_rng()

        class Data:
            # fake 8-school output
            obj = {}
            for key, shape in {"mu": [], "tau": [], "eta": [8], "theta": [8]}.items():
                obj[key] = rng.normal(size=(chains, draws, *shape))

        return Data

    def get_datatree(self, data):
        return convert_to_datatree(
            data.obj,
            group="posterior",
            coords={"school": np.arange(8)},
            dims={"theta": ["school"], "eta": ["school"]},
        )

    def check_var_names_coords_dims(self, dataset):
        assert set(dataset.data_vars) == {"mu", "tau", "eta", "theta"}
        assert set(dataset.coords) == {"chain", "draw", "school"}

    def test_convert_to_datatree(self, data):
        data = self.get_datatree(data)
        assert "/posterior" in data.groups
        self.check_var_names_coords_dims(data.posterior)

    def test_convert_to_dataset(self, draws, chains, data):
        dataset = convert_to_dataset(
            data.obj,
            coords={"school": np.arange(8)},
            dims={"theta": ["school"], "eta": ["school"]},
        )
        assert dataset.draw.shape == (draws,)
        assert dataset.chain.shape == (chains,)
        assert dataset.school.shape == (8,)
        assert dataset.theta.shape == (chains, draws, 8)


class TestExtract:
    def test_default(self, centered_eight, chains, draws):
        post = extract(centered_eight)
        assert isinstance(post, xr.Dataset)
        assert "sample" in post.sizes
        assert post.theta.size == (chains * draws * 8)

    def test_seed(self, centered_eight):
        post = extract(centered_eight, random_seed=7)
        post_pred = extract(centered_eight, group="posterior_predictive", random_seed=7)
        assert all(post.sample == post_pred.sample)

    def test_no_combine(self, centered_eight, chains, draws):
        post = extract(centered_eight, combined=False)
        assert "sample" not in post.sizes
        assert post.sizes["chain"] == chains
        assert post.sizes["draw"] == draws

    def test_single_sample_dims(self, centered_eight):
        post = extract(centered_eight, sample_dims="draw")
        xr.testing.assert_equal(post, centered_eight.posterior.to_dataset())

    def test_var_name_group(self, centered_eight):
        prior = extract(centered_eight, group="prior", var_names="the", filter_vars="like")
        assert {} == prior.attrs
        assert "theta" in prior.name

    def test_return_as_dataset(self, centered_eight):
        prior = extract(
            centered_eight,
            group="prior",
            var_names="the",
            filter_vars="like",
            return_as_dataset=True,
        )
        assert prior.attrs == centered_eight.prior.attrs
        assert "theta" in prior.data_vars
        assert "mu" not in prior.data_vars

    def test_subset_samples(self, centered_eight):
        post = extract(centered_eight, num_samples=10)
        assert post.sizes["sample"] == 10
        assert post.attrs == centered_eight.posterior.attrs

    def test_subset_single_sample_dims(self, centered_eight):
        post = extract(centered_eight, sample_dims="draw", num_samples=4)
        assert post.sizes["draw"] == 4
        assert post.attrs == centered_eight.posterior.attrs

    def test_dataarray_return(self, centered_eight):
        post = extract(centered_eight.posterior["theta"])
        assert isinstance(post, xr.DataArray)
        post = extract(centered_eight.posterior.to_dataset()[["theta"]])
        assert isinstance(post, xr.DataArray)
        post = extract(centered_eight, var_names="theta")
        assert isinstance(post, xr.DataArray)

    def test_weights(self, centered_eight):
        rng = np.random.default_rng()
        weights = rng.random(
            centered_eight.posterior.sizes["chain"] * centered_eight.posterior.sizes["draw"]
        )
        weights[:10] = 0
        weight_0_idxs = [(0, i) for i in range(10)]
        weights /= weights.sum()
        post = extract(centered_eight, num_samples=len(weights), weights=weights)
        assert post.sizes["sample"] == len(weights)
        assert not any(idx in list(post.sample.to_numpy()) for idx in weight_0_idxs)
        assert post.attrs == centered_eight.posterior.attrs
