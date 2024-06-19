"""emcee-specific conversion code."""

import warnings

import numpy as np
from datatree import DataTree

from .base import dict_to_dataset
from .rcparams import rc_context


def _verify_names(sampler, var_names, arg_names, slices):
    """Make sure var_names and arg_names are assigned reasonably.

    This is meant to run before loading emcee objects into DataTree.
    In case var_names or arg_names is None, will provide defaults. If they are
    not None, it verifies there are the right number of them.

    Throws a ValueError in case validation fails.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Fitted emcee sampler
    var_names : list of str or None
        Names for the emcee parameters
    arg_names : list of str or None
        Names for the args/observations provided to emcee
    slices : list or None
        slices to select the variables (used for multidimensional variables).
        It can be a list of ``slice`` objects or lists/arrays to be used for indexing.

    Returns
    -------
    list of str, list of str, list
        Defaults for var_names, arg_names and slices
    """
    if hasattr(sampler, "log_prob_fn"):
        ndim = sampler.get_chain().shape[-1]
        num_args = len(sampler.log_prob_fn.args)
    else:
        # when saving to an hdf5 file, only the samples are stored
        ndim = sampler.get_chain().shape[-1]
        num_args = 0

    if slices is None:
        slices = np.arange(ndim)
        num_vars = ndim
    else:
        num_vars = len(slices)
    indices = np.arange(ndim)
    slicing_try = np.concatenate([np.atleast_1d(indices[idx]) for idx in slices])
    if len(set(slicing_try)) != ndim:
        warnings.warn(
            "Check slices: Not all parameters in chain captured. "
            f"{ndim} are present, and {len(slicing_try)} have been captured.",
            UserWarning,
        )
    if len(slicing_try) != len(set(slicing_try)):
        warnings.warn(f"Overlapping slices. Check the index present: {slicing_try}", UserWarning)

    if var_names is None:
        var_names = [f"var_{idx}" for idx in range(num_vars)]
    if arg_names is None:
        arg_names = [f"arg_{idx}" for idx in range(num_args)]

    if len(var_names) != num_vars:
        raise ValueError(
            f"The sampler has {num_vars} variables, "
            f"but only {len(var_names)} var_names were provided!"
        )

    if len(arg_names) != num_args:
        raise ValueError(
            f"The sampler has {num_args} args, "
            f"but only {len(arg_names)} arg_names were provided!"
        )
    return var_names, arg_names, slices


# pylint: disable=too-many-instance-attributes
class EmceeConverter:
    """Encapsulate emcee specific logic."""

    def __init__(
        self,
        sampler,
        var_names=None,
        slices=None,
        arg_names=None,
        arg_groups=None,
        blob_names=None,
        blob_groups=None,
        index_origin=None,
        coords=None,
        dims=None,
        check_conventions=True,
    ):
        var_names, arg_names, slices = _verify_names(sampler, var_names, arg_names, slices)
        self.sampler = sampler
        self.var_names = var_names
        self.slices = slices
        self.arg_names = arg_names
        self.arg_groups = arg_groups
        self.blob_names = blob_names
        self.blob_groups = blob_groups
        self.index_origin = index_origin
        self.coords = coords
        self.dims = dims
        self.check_conventions = check_conventions
        import emcee

        self.emcee = emcee

    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        # Use emcee3 syntax, else use emcee2
        samples_ary = self.sampler.get_chain()

        data = {
            var_name: (samples_ary[(..., idx)])
            for idx, var_name in zip(self.slices, self.var_names)
        }
        return dict_to_dataset(
            data,
            inference_library=self.emcee,
            coords=self.coords,
            dims=self.dims,
            index_origin=self.index_origin,
            check_conventions=self.check_conventions,
        )

    def args_to_xarray(self):
        """Convert emcee args to observed and constant_data xarray Datasets."""
        if self.arg_groups is None:
            self.arg_groups = ["observed_data" for _ in self.arg_names]
        if len(self.arg_names) != len(self.arg_groups):
            raise ValueError(
                "arg_names and arg_groups must have the same length, or arg_groups be None"
            )
        arg_groups_set = set(self.arg_groups)
        bad_groups = [
            group for group in arg_groups_set if group not in ("observed_data", "constant_data")
        ]
        if bad_groups:
            raise SyntaxError(
                "all arg_groups values should be either 'observed_data' or 'constant_data' , "
                f"not {bad_groups}"
            )
        obs_const_dict = {group: {} for group in arg_groups_set}
        for idx, (arg_name, group) in enumerate(zip(self.arg_names, self.arg_groups)):
            obs_const_dict[group][arg_name] = np.atleast_1d(self.sampler.log_prob_fn.args[idx])
        for group, datadict in obs_const_dict.items():
            obs_const_dict[group] = dict_to_dataset(
                datadict,
                inference_library=self.emcee,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
                sample_dims=[],
                check_conventions=self.check_conventions,
            )
        return obs_const_dict

    def blobs_to_dict(self):
        """Convert blobs to dictionary {groupname: xr.Dataset}.

        It also stores lp values in sample_stats group.
        """
        store_blobs = self.blob_names is not None
        self.blob_names = [] if self.blob_names is None else self.blob_names
        if self.blob_groups is None:
            self.blob_groups = ["log_likelihood" for _ in self.blob_names]
        if len(self.blob_names) != len(self.blob_groups):
            raise ValueError(
                "blob_names and blob_groups must have the same length, or blob_groups be None"
            )
        if store_blobs:
            blobs = self.sampler.get_blobs()
            if (blobs is None or blobs.size == 0) and self.blob_names:
                raise ValueError("No blobs in sampler, blob_names must be None")
            if len(blobs.shape) == 2:
                blobs = np.expand_dims(blobs, axis=-1)
            ndraws, nwalkers, nblobs, *_ = blobs.shape
            if len(self.blob_names) != nblobs and len(self.blob_names) > 1:
                raise ValueError(
                    "Incorrect number of blob names. "
                    f"Expected {nblobs}, found {len(self.blob_names)}"
                )
        blob_groups_set = set(self.blob_groups)
        blob_groups_set.add("sample_stats")
        idata_groups = ("posterior", "observed_data", "constant_data")
        if np.any(np.isin(list(blob_groups_set), idata_groups)):
            raise SyntaxError(
                f"{idata_groups} groups should not come from blobs. "
                "Using them here would overwrite their actual values"
            )
        blob_dict = {group: {} for group in blob_groups_set}
        if len(self.blob_names) == 1:
            blob_dict[self.blob_groups[0]][self.blob_names[0]] = blobs
        else:
            for i_blob, (name, group) in enumerate(zip(self.blob_names, self.blob_groups)):
                # for coherent blobs (all having the same dimensions) one line is enough
                blob = blobs[:, :, i_blob]
                # for blobs of different size, we get an array of arrays, which we convert
                # to an ndarray per blob_name
                if blob.dtype == object:
                    blob = blob.reshape(-1)
                    blob = np.stack(blob)
                    blob = blob.reshape((ndraws, nwalkers, -1))
                blob_dict[group][name] = np.squeeze(blob)

        # store lp in sample_stats group
        blob_dict["sample_stats"]["lp"] = self.sampler.get_log_prob()
        for key, values in blob_dict.items():
            blob_dict[key] = dict_to_dataset(
                values,
                inference_library=self.emcee,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
            )
        return blob_dict

    def to_datatree(self):
        """Convert all available data to an InferenceData object."""
        blobs_dict = self.blobs_to_dict()
        obs_const_dict = self.args_to_xarray()
        return DataTree.from_dict(
            {"posterior": self.posterior_to_xarray(), **obs_const_dict, **blobs_dict}
        )


def from_emcee(
    sampler=None,
    var_names=None,
    slices=None,
    arg_names=None,
    arg_groups=None,
    blob_names=None,
    blob_groups=None,
    index_origin=None,
    coords=None,
    dims=None,
    check_conventions=True,
):
    """Convert emcee data into a DataTree.

    For a usage example read :ref:`emcee_conversion`


    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Fitted sampler from emcee.
    var_names : list of str, optional
        A list of names for variables in the sampler
    slices : list of array-like or slice, optional
        A list containing the indexes of each variable. Should only be used
        for multidimensional variables.
    arg_names : list of str, optional
        A list of names for args in the sampler
    arg_groups : list of str, optional
        A list of the group names (either ``observed_data`` or ``constant_data``) where
        args in the sampler are stored. If None, all args will be stored in observed
        data group.
    blob_names : list of str, optional
        A list of names for blobs in the sampler. When None,
        blobs are omitted, independently of them being present
        in the sampler or not.
    blob_groups : list of str, optional
        A list of the groups where blob_names variables
        should be assigned respectively. If blob_names!=None
        and blob_groups is None, all variables are assigned
        to log_likelihood group
    coords : dict of {str : array_like}, optional
        Map of dimensions to coordinates
    dims : dict of {str : list of str}, optional
        Map variable names to their coordinates
    check_conventions : bool, default True
        Check ArviZ conventions when converting to DataTree

    Returns
    -------
    DataTree

    """
    with rc_context(rc={"data.sample_dims": ["draw", "chain"]}):
        return EmceeConverter(
            sampler=sampler,
            var_names=var_names,
            slices=slices,
            arg_names=arg_names,
            arg_groups=arg_groups,
            blob_names=blob_names,
            blob_groups=blob_groups,
            index_origin=index_origin,
            coords=coords,
            dims=dims,
            check_conventions=check_conventions,
        ).to_datatree()
