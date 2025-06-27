# pylint: disable=no-member, no-self-use, redefined-outer-name
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arviz_base import dataset_to_dataarray, dataset_to_dataframe, extract, references_to_dataset
from arviz_base.labels import DimCoordLabeller


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
        assert "theta" == prior.name

    def test_keep_dataset(self, centered_eight):
        prior = extract(
            centered_eight, group="prior", var_names="the", filter_vars="like", keep_dataset=True
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


class TestDsToDa:
    def test_default(self, centered_eight):
        post_ds = centered_eight.posterior.dataset
        post_da = dataset_to_dataarray(post_ds)
        assert isinstance(post_da, xr.DataArray)
        assert "chain" in post_da.dims
        assert post_da.sizes["chain"] == post_ds.sizes["chain"]
        assert "draw" in post_da.dims
        assert post_da.sizes["draw"] == post_ds.sizes["draw"]
        assert "label" in post_da.dims
        assert "school" not in post_da.dims
        assert post_da.sizes["label"] == 10
        assert "mu" in post_da.coords["label"].to_numpy()
        assert "theta[Choate]" in post_da.coords["label"].to_numpy()
        assert "theta" in post_da.coords["variable"].to_numpy()
        assert "mu" in post_da.coords["variable"].to_numpy()

    def test_sample_dims(self, centered_eight):
        post_ds = centered_eight.posterior.dataset
        post_da = dataset_to_dataarray(post_ds, sample_dims=["draw"])
        assert isinstance(post_da, xr.DataArray)
        assert "draw" in post_da.dims
        assert post_da.sizes["draw"] == post_ds.sizes["draw"]
        assert "label" in post_da.dims
        assert "chain" not in post_da.dims
        assert "school" not in post_da.dims
        assert post_da.sizes["label"] == 10 * post_ds.sizes["chain"]
        assert "mu[0]" in post_da["label"].to_numpy()
        assert "theta[0, Choate]" in post_da["label"].to_numpy()
        assert "theta" in post_da.coords["variable"].to_numpy()
        assert "mu" in post_da.coords["variable"].to_numpy()

    def test_stacked(self, centered_eight):
        post_ds = centered_eight.posterior.to_dataset().stack(sample=["chain", "draw"])
        post_da = dataset_to_dataarray(post_ds, sample_dims=["sample"])
        assert isinstance(post_da, xr.DataArray)
        assert "sample" in post_da.dims
        assert post_da.sizes["sample"] == post_ds.sizes["sample"]
        assert "label" in post_da.dims
        assert "school" not in post_da.dims
        assert "chain" not in post_da.dims
        assert "draw" not in post_da.dims
        assert post_da.sizes["label"] == 10
        assert "mu" in post_da["label"].to_numpy()
        assert "theta[Choate]" in post_da["label"].to_numpy()
        assert "theta" in post_da.coords["variable"].to_numpy()
        assert "mu" in post_da.coords["variable"].to_numpy()

    def test_labeller(self, centered_eight):
        post_ds = centered_eight.posterior.dataset
        post_da = dataset_to_dataarray(post_ds, labeller=DimCoordLabeller())
        assert isinstance(post_da, xr.DataArray)
        assert "label" in post_da.dims
        assert post_da.sizes["label"] == 10
        assert "mu" in post_da.coords["label"].to_numpy()
        assert "theta[school: Choate]" in post_da.coords["label"].to_numpy()
        assert "theta" in post_da.coords["variable"].to_numpy()

    def test_no_coords(self, centered_eight):
        post_ds = centered_eight.posterior.dataset
        post_da = dataset_to_dataarray(post_ds, add_coords=False)
        assert isinstance(post_da, xr.DataArray)
        assert "label" in post_da.dims
        assert post_da.sizes["label"] == 10
        assert "mu" in post_da.coords["label"].to_numpy()
        assert "theta[Choate]" in post_da.coords["label"].to_numpy()
        assert "variable" not in post_da.coords
        assert "school" not in post_da.coords

    def test_new_dim(self, centered_eight):
        post_ds = centered_eight.posterior.dataset
        post_da = dataset_to_dataarray(post_ds, new_dim="new_dim")
        assert isinstance(post_da, xr.DataArray)
        assert "new_dim" in post_da.dims
        assert post_da.sizes["new_dim"] == 10
        assert "mu" in post_da.coords["new_dim"].to_numpy()
        assert "theta[Choate]" in post_da.coords["new_dim"].to_numpy()


class TestDsToDf:
    @pytest.mark.parametrize("multiindex", [True, False, "row", "column"])
    def test_df_specific(self, centered_eight, multiindex):
        """Test dataset_to_dataframe specific behaviour and args.

        Everything else is delegated to `dataset_to_dataarray` and its tests.
        """
        post_ds = centered_eight.posterior.dataset
        post_df = dataset_to_dataframe(post_ds, multiindex=multiindex)
        assert isinstance(post_df, pd.DataFrame)
        assert post_df.shape == (post_ds.sizes["chain"] * post_ds.sizes["draw"], 10)
        if multiindex is True or multiindex == "row":
            assert isinstance(post_df.index, pd.MultiIndex)
        else:
            assert not isinstance(post_df.index, pd.MultiIndex)
        if multiindex is True or multiindex == "column":
            assert isinstance(post_df.columns, pd.MultiIndex)
        else:
            assert not isinstance(post_df.columns, pd.MultiIndex)


class TestRefToDs:
    def test_ds(self, centered_eight):
        ds_in = centered_eight.posterior.dataset.mean(["chain", "draw"])
        ds_out = references_to_dataset(ds_in, xr.Dataset(), None)
        assert ds_in is ds_out

    def test_scalar(self, centered_eight):
        post_ds = centered_eight.posterior.dataset
        ref_ds = references_to_dataset(0, post_ds, sample_dims=["chain", "draw"])
        assert isinstance(ref_ds, xr.Dataset)
        assert all(var_name in ref_ds.data_vars for var_name in ("mu", "theta", "tau"))
        assert "chain" not in ref_ds.dims
        assert "draw" not in ref_ds.dims
        assert "ref_dim" not in ref_ds.dims
        assert "school" in ref_ds.dims

    def test_array(self, centered_eight):
        post_ds = centered_eight.posterior.dataset
        ref_ds = references_to_dataset(np.array([-1, 0, 1]), post_ds, sample_dims=["chain", "draw"])
        assert isinstance(ref_ds, xr.Dataset)
        assert all(var_name in ref_ds.data_vars for var_name in ("mu", "theta", "tau"))
        assert "chain" not in ref_ds.dims
        assert "draw" not in ref_ds.dims
        assert "school" in ref_ds.dims
        assert "ref_dim" in ref_ds.dims
        assert ref_ds.sizes["ref_dim"] == 3

    def test_2darray(self, centered_eight):
        post_ds = centered_eight.posterior.dataset
        ref_ds = references_to_dataset(
            np.array([[-1, 0, 1], [0, 1, 2]]), post_ds, sample_dims=["chain", "draw"]
        )
        assert isinstance(ref_ds, xr.Dataset)
        assert all(var_name in ref_ds.data_vars for var_name in ("mu", "theta", "tau"))
        assert "chain" not in ref_ds.dims
        assert "draw" not in ref_ds.dims
        assert "school" in ref_ds.dims
        assert "ref_dim_0" in ref_ds.dims
        assert ref_ds.sizes["ref_dim_0"] == 2
        assert "ref_dim_1" in ref_ds.dims
        assert ref_ds.sizes["ref_dim_1"] == 3

    def test_dict(self, centered_eight):
        post_ds = centered_eight.posterior.dataset
        ref_ds = references_to_dataset(
            {"mu": np.array([-1, 0, 1]), "theta": 0}, post_ds, sample_dims=["chain", "draw"]
        )
        assert isinstance(ref_ds, xr.Dataset)
        assert all(var_name in ref_ds.data_vars for var_name in ("mu", "theta"))
        assert "tau" not in ref_ds.data_vars
        assert "chain" not in ref_ds.dims
        assert "draw" not in ref_ds.dims
        assert "school" in ref_ds.dims
        assert "ref_dim" in ref_ds.dims
        assert ref_ds.sizes["ref_dim"] == 3
        assert not np.any(np.isnan(ref_ds["mu"]))
        assert np.allclose(ref_ds["theta"].isel(ref_dim=0), 0)
        assert np.all(np.isnan(ref_ds["theta"].isel(ref_dim=[1, 2])))
