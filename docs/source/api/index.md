# API reference

## User facing converters

```{eval-rst}
.. autosummary::
   :toctree: generated/

   arviz_base.convert_to_datatree
   arviz_base.extract
   arviz_base.from_dict
```

## Library specific converters

```{eval-rst}
.. autosummary::
   :toctree: generated/

   arviz_base.from_cmdstanpy
   arviz_base.from_emcee
```

More coming soon...

## Iteration and subsetting

```{eval-rst}
.. autosummary::
   :toctree: generated/

   arviz_base.xarray_sel_iter
   arviz_base.xarray_var_iter
```


## Example datasets
The behaviour of the functions in this section is partially controlled by the
following environment variable:

:::{envvar} ARVIZ_DATA
If present, store remote datasets after downloading in the location indicated there.
Otherwise, datasets are stored at `~/arviz_data/`
:::

```{eval-rst}
.. autosummary::
   :toctree: generated/

   arviz_base.load_arviz_data
   arviz_base.list_datasets
   arviz_base.get_data_home
   arviz_base.clear_data_home
```

## Conversion utilities

```{eval-rst}
.. autosummary::
   :toctree: generated/

   arviz_base.convert_to_dataset
   arviz_base.dict_to_dataset
   arviz_base.generate_dims_coords
   arviz_base.make_attrs
   arviz_base.ndarray_to_dataarray
```

## Configuration
Most ArviZ default values are regulated by {class}`arviz_base.rcParams`, a class similar
to a dictionary storing key-value pairs inspired by the one in matplotlib.
It is similar to a dictionary and not a dictionary though because all keys are fixed,
and each key has associated a validation function to help prevent setting nonsensical defaults.

### ArviZ configuration file

The `rcParams` class is generated and populated at import time. ArviZ checks several
locations for a file named `arvizrc` and, if found, prefers those settings over the library ones.

The locations checked are the following:

1. Current working directory, {func}`os.getcwd`
1. Location indicated by {envvar}`ARVIZ_DATA` environment variable
1. The third and last location checked is OS dependent:
   * On Linux: `$XDG_CONFIG_HOME/arviz` if exists, otherwise `~/.config/arviz/`
   * Elsewhere: `~/.arviz/`

:::{dropdown} Example `arvizrc` file
:name: arvizrc
:open:

```none
data.index_origin : 1
plot.backend : bokeh
stats.ci_kind : hdi
stats.ci_prob : 0.95
```

All available keys are listed below. The `arvizrc` file can have any subset of the keys,
it isn't necessary to include them all. For those keys without a user defined default,
the library one is used.
:::

### Context manager
A context manager is also available to temporarily change the default settings.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   arviz_base.rc_context
```

## rcParams
Below, all keys available within `rcParams` are listed, along with their library default.

Keys can be accessed or modified via ``arviz_base.rcParams[key]``, for example,
``arviz_base.rcParams["data.sample_dims"]``.

:::{important}
These defaults are subject to change. If you rely on a specific default value,
you should create an {ref}`arvizrc <arvizrc>` file with the key-value pairs you rely on.

The goal of the ArviZ team is to try and adapt to best practices as they evolve,
which sometimes requires updating default values, for example to use new algorithms.
:::

### data


```{eval-rst}
.. py:data:: data.http_protocol
    :type: str
    :value: "https"

    Protocol for loading remote datasets. Can be "https" or "http".

.. py:data:: data.index_origin
    :type: int
    :value: 0

    Index origin. By default, ArviZ adds coordinate values to all dimensions.
    If no coordinate values were provided, ArviZ generates integer indices
    as coordinate values starting at `index_origin`.

.. py:data:: data.log_likelihood
    :type: bool
    :value: True

    Whether to save pointwise log likelihood values.

.. py:data:: data.sample_dims
    :type: list
    :value: ["chain", "draw"]

    What the sampling dimensions are named. These are the dimensions that will be
    reduced by default when computing or plotting, therefore, they should be always present.

.. py:data:: data.save_warmup
    :type: bool
    :value: False

    Whether to save warmup iterations.
```

### stats

```{eval-rst}
.. py:data:: stats.module
    :type: str
    :value: "arviz_stats.base"

    Preferred module for stats computations

.. py:data:: stats.ci_kind
    :type: str
    :value: "eti"

    Type of credible interval to compute by default, one of "eti" or "hdi".

.. py:data:: stats.ci_prob
    :type: float
    :value: 0.83

    The default probability of computed credible intervals. Its default value here
    is also a friendly reminder of the arbitrary nature of commonly values like 95%

.. py:data:: stats.information_criterion
    :type: str
    :value: "loo"

    Default algorithm for predictive performance quantification, one of "loo" or "waic".

.. py:data:: stats.ic_compare_method
    :type: str
    :value: "stacking"

    Method for comparing multiple models using their information criteria values,
    one of "stacking", "bb-pseudo-bma" or "pseudo-mba".

.. py:data:: stats.ic_pointwise
    :type: bool
    :value: True

    Whether to return pointwise values when computing the
    :data:`information criterion <stats.information_criterion>`.

.. py:data:: stats.ic_scale
    :type: str
    :value: "log"

    The scale in which to return
    :data:`information criterion <stats.information_criterion>` values,
    one of "deviance" (common in the past and reason of the information criterion naming),
    "log" or "negative_log".
```

### plots

```{eval-rst}
.. py:data:: plot.backend
    :type: str
    :value: "matplotlib"

    Default plotting backend for :mod:`arviz_plots`, one of "matplotlib", "bokeh" or "none".

.. py:data:: plot.density_kind
    :type: str
    :value: "kde"

    Default representation for 1D marginal densities, one of "kde", "hist", "ecdf" or "dot".

.. py:data:: plot.max_subplots
    :type: int
    :value: 40

    Maximum number of :term:`arviz_plots:plots` that can be generated at once.

.. py:data:: plot.point_estimate
    :type: str
    :value: "mean"

    Default statistical summary for centrality, one of "mean", "median" or "mode".
```
