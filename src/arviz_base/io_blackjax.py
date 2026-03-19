"""BlackJAX-specific conversion code."""

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any

import lazy_loader as _lazy
import numpy as np
from xarray import DataTree

from arviz_base.base import dict_to_dataset, requires
from arviz_base.rcparams import rc_context, rcParams

if TYPE_CHECKING:
    import jax
else:
    jax = _lazy.load("jax")

__all__ = ["from_blackjax"]

_NUTS_STAT_RENAME = {
    "energy": "energy",
    "num_integration_steps": "n_steps",
    "acceptance_rate": "acceptance_rate",
    "is_divergent": "diverging",
    "step_size": "step_size",
}


def _jax_to_numpy(val: Any) -> np.ndarray:
    """Convert a JAX array (or any array-like) to a numpy array."""
    return np.asarray(jax.device_get(val))


def _position_to_dict(position: Any) -> dict[str, np.ndarray]:
    """Normalise a BlackJAX position pytree to a plain dict of numpy arrays.

    BlackJAX positions can be a dict, a NamedTuple, or a bare array.
    Bare arrays are stored under the variable name ``"x"``.
    """
    if isinstance(position, dict):
        return {k: _jax_to_numpy(v) for k, v in position.items()}
    if hasattr(position, "_fields"):  # NamedTuple
        return {f: _jax_to_numpy(getattr(position, f)) for f in position._fields}
    return {"x": _jax_to_numpy(position)}


def _ensure_chain_dim(
    samples: dict[str, np.ndarray], num_chains: int | None
) -> dict[str, np.ndarray]:
    """Ensure every array has a leading chain dimension.

    Single-chain BlackJAX output has shape ``(n_draws, *event_shape)``.
    Multi-chain output (via ``jax.vmap`` / ``jax.pmap``) has shape
    ``(n_chains, n_draws, *event_shape)``.  A chain dim of size 1 is inserted
    for single-chain runs.
    """
    result: dict[str, np.ndarray] = {}
    for name, arr in samples.items():
        if arr.ndim == 0:
            result[name] = arr.reshape(1, 1)
        elif num_chains is not None and arr.ndim >= 2 and arr.shape[0] == num_chains:
            result[name] = arr
        else:
            result[name] = np.expand_dims(arr, axis=0)
    return result


def _extract_info_fields(info: Any, rename_map: dict[str, str]) -> dict[str, np.ndarray]:
    """Pull known stat fields out of a BlackJAX info object.

    Works with both dict-like and attribute-style info objects and returns
    a plain dict of numpy arrays using the names in *rename_map*.
    """
    data: dict[str, np.ndarray] = {}
    if info is None:
        return data
    for src_name, dst_name in rename_map.items():
        val = info.get(src_name) if isinstance(info, dict) else getattr(info, src_name, None)
        if val is not None:
            data[dst_name] = _jax_to_numpy(val)
    return data


class BlackJAXConverter:
    """Encapsulate BlackJAX-specific conversion logic."""

    def __init__(
        self,
        *,
        posterior: Any = None,
        prior: dict[str, Any] | None = None,
        info: Any = None,
        posterior_predictive: dict[str, Any] | None = None,
        observed_data: dict[str, Any] | None = None,
        constant_data: dict[str, Any] | None = None,
        coords: dict[str, Any] | None = None,
        dims: dict[Hashable, Sequence[Hashable]] | None = None,
        index_origin: int | None = None,
        num_chains: int | None = None,
        max_tree_depth: int | None = None,
    ) -> None:
        self.index_origin = rcParams["data.index_origin"] if index_origin is None else index_origin
        self.coords = coords
        self.dims = dims
        self.num_chains = num_chains
        self.max_tree_depth = max_tree_depth

        if posterior is not None:
            raw = _position_to_dict(posterior)
            self._samples: dict[str, np.ndarray] | None = _ensure_chain_dim(raw, num_chains)
        else:
            self._samples = None

        stats = _extract_info_fields(info, _NUTS_STAT_RENAME) if info is not None else None
        self._stats: dict[str, np.ndarray] | None = (
            _ensure_chain_dim(stats, num_chains) if stats else None
        )

        self.posterior_predictive: dict[str, np.ndarray] | None = (
            {k: _jax_to_numpy(v) for k, v in posterior_predictive.items()}
            if posterior_predictive is not None
            else None
        )
        self.prior: dict[str, np.ndarray] | None = (
            {k: _jax_to_numpy(v) for k, v in prior.items()} if prior is not None else None
        )
        self.observed_data = observed_data
        self.constant_data = constant_data

    @requires("_samples")
    def posterior_to_xarray(self):
        """Convert posterior samples to an xarray Dataset."""
        return dict_to_dataset(
            self._samples,  # type: ignore[arg-type]
            coords=self.coords,
            dims=self.dims,
            index_origin=self.index_origin,
        )

    @requires("_stats")
    def sample_stats_to_xarray(self):
        """Convert BlackJAX info fields to a sample_stats xarray Dataset."""
        data: dict[str, np.ndarray] = dict(self._stats)  # type: ignore[arg-type]
        if "n_steps" in data:
            data["tree_depth"] = np.log2(data["n_steps"]).astype(int) + 1
            if self.max_tree_depth is not None:
                data["reached_max_tree_depth"] = data["tree_depth"] >= self.max_tree_depth
        return dict_to_dataset(
            data,
            coords=self.coords,
            dims=None,
            index_origin=self.index_origin,
        )

    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior predictive samples to an xarray Dataset."""
        return dict_to_dataset(
            self.posterior_predictive,  # type: ignore[arg-type]
            coords=self.coords,
            dims=self.dims,
            index_origin=self.index_origin,
        )

    def priors_to_xarray(self) -> dict[str, Any]:
        """Convert prior samples to xarray Datasets.

        If ``posterior`` is also present, variables that appear in the posterior
        are placed in the ``prior`` group; any remaining variables (i.e. those
        not sampled during inference) are placed in ``prior_predictive``.
        When no posterior is available all prior variables go into ``prior``.
        """
        if self.prior is None:
            return {"prior": None, "prior_predictive": None}

        if self._samples is not None:
            posterior_vars = set(self._samples.keys())
            prior_vars = {k: v for k, v in self.prior.items() if k in posterior_vars}
            prior_predictive_vars = {k: v for k, v in self.prior.items() if k not in posterior_vars}
        else:
            prior_vars = self.prior
            prior_predictive_vars = {}

        prior_ds = (
            dict_to_dataset(
                prior_vars,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
            )
            if prior_vars
            else None
        )
        prior_predictive_ds = (
            dict_to_dataset(
                prior_predictive_vars,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
            )
            if prior_predictive_vars
            else None
        )
        return {"prior": prior_ds, "prior_predictive": prior_predictive_ds}

    @requires("observed_data")
    def observed_data_to_xarray(self):
        """Convert observed data to an xarray Dataset."""
        return dict_to_dataset(
            self.observed_data,  # type: ignore[arg-type]
            coords=self.coords,
            dims=self.dims,
            sample_dims=[],
            index_origin=self.index_origin,
        )

    @requires("constant_data")
    def constant_data_to_xarray(self):
        """Convert constant data to an xarray Dataset."""
        return dict_to_dataset(
            self.constant_data,  # type: ignore[arg-type]
            coords=self.coords,
            dims=self.dims,
            sample_dims=[],
            index_origin=self.index_origin,
        )

    def to_datatree(self) -> DataTree:
        """Convert all available data to a DataTree object."""
        groups: dict[str, Any] = {
            "posterior": self.posterior_to_xarray(),
            "sample_stats": self.sample_stats_to_xarray(),
            "posterior_predictive": self.posterior_predictive_to_xarray(),
            **self.priors_to_xarray(),
            "observed_data": self.observed_data_to_xarray(),
            "constant_data": self.constant_data_to_xarray(),
        }
        return DataTree.from_dict({g: ds for g, ds in groups.items() if ds is not None})


def from_blackjax(
    posterior: Any = None,
    *,
    prior: dict[str, Any] | None = None,
    info: Any = None,
    posterior_predictive: dict[str, Any] | None = None,
    observed_data: dict[str, Any] | None = None,
    constant_data: dict[str, Any] | None = None,
    coords: dict[str, Any] | None = None,
    dims: dict[Hashable, Sequence[Hashable]] | None = None,
    index_origin: int | None = None,
    num_chains: int | None = None,
    max_tree_depth: int | None = None,
) -> DataTree:
    """Convert BlackJAX sampling output into a DataTree object.

    Parameters
    ----------
    posterior : pytree, optional
        The ``.position`` field of a BlackJAX state object.  Can be a dict,
        a NamedTuple, or a bare JAX array.  Single-chain shape is expected to
        be ``(n_draws, *event_shape)``; multi-chain (after ``jax.vmap`` /
        ``jax.pmap``) should be ``(n_chains, n_draws, *event_shape)``.
    prior : dict, optional
        Prior samples as a dictionary mapping variable names to arrays.  If a
        variable name also appears in ``posterior``, it is placed in the
        ``prior`` group; otherwise it goes into ``prior_predictive``.
    info : pytree, optional
        The info object returned alongside the BlackJAX state (second element
        of the ``(state, info)`` tuple).  For NUTS this contains fields such
        as ``num_integration_steps``, ``is_divergent``, ``acceptance_rate``,
        ``energy``, and ``step_size``, which are mapped to ArviZ standard
        ``sample_stats`` names.
    posterior_predictive : dict, optional
        Dictionary of posterior predictive samples.
    observed_data : dict, optional
        Dictionary of observed data variables.
    constant_data : dict, optional
        Dictionary of constant data variables.
    coords : dict, optional
        Map of dimension names to coordinate values.
    dims : dict of {str: list of str}, optional
        Map of variable names to their dimension names.
    index_origin : int, optional
        Starting index for integer coordinates.  Defaults to
        ``rcParams["data.index_origin"]``.
    num_chains : int, optional
        Number of chains.  Required when the position arrays do not already
        have a leading chain dimension (i.e. single-chain runs where you want
        explicit chain tracking, or when arrays are flattened).
    max_tree_depth : int, optional
        Maximum tree depth used during NUTS sampling.  When provided,
        ``reached_max_tree_depth`` is added to ``sample_stats``.

    Returns
    -------
    DataTree

    Examples
    --------
    Single-chain NUTS run:

    .. code-block:: python

        import jax
        import blackjax

        rng_key = jax.random.PRNGKey(0)

        nuts = blackjax.nuts(log_prob, step_size=1e-3, inverse_mass_matrix=jnp.ones(2))
        state = nuts.init(initial_position)

        def one_step(state, rng_key):
            state, info = nuts.step(rng_key, state)
            return state, (state, info)

        keys = jax.random.split(rng_key, num_samples)
        _, (states, infos) = jax.lax.scan(one_step, state, keys)

        idata = from_blackjax(posterior=states.position, info=infos)

    With prior samples:

    .. code-block:: python

        prior_samples = {"mu": prior_mu, "tau": prior_tau, "y_hat": prior_y}
        idata = from_blackjax(
            posterior=states.position,
            prior=prior_samples,
        )
        # mu and tau appear in idata.prior
        # y_hat appears in idata.prior_predictive

    Multi-chain run via ``jax.vmap``:

    .. code-block:: python

        idata = from_blackjax(
            posterior=states.position,
            info=infos,
            num_chains=n_chains,
            max_tree_depth=10,
        )
    """
    with rc_context(rc={"data.sample_dims": ["chain", "draw"]}):
        return BlackJAXConverter(
            posterior=posterior,
            prior=prior,
            info=info,
            posterior_predictive=posterior_predictive,
            observed_data=observed_data,
            constant_data=constant_data,
            coords=coords,
            dims=dims,
            index_origin=index_origin,
            num_chains=num_chains,
            max_tree_depth=max_tree_depth,
        ).to_datatree()
