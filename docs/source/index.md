# arviz-base
ArviZ base features and converters.

## Installation

It currently can only be installed with pip:

::::{tab-set}
:::{tab-item} PyPI
:sync: stable

```bash
pip install arviz-base
```
:::
:::{tab-item} GitHub
:sync: dev

```bash
pip install arviz-base @ git+https://github.com/arviz-devs/arviz-base
```
:::
::::

Note that `arviz-base` is a minimal package, which only depends on
xarray NumPy and typing-extensions.

Everything else (netcdf, zarr, dask...) are optional dependencies.
This allows installing only those that are needed, e.g. if you
only plan to use zarr, there is no need to install netcdf.

For convenience, some bundles are available to be installed with:

::::{tab-set}
:::{tab-item} PyPI
:sync: stable

```bash
pip install "arviz-base[<option>]"
```
:::
:::{tab-item} GitHub
:sync: dev

```bash
pip install "arviz-base[<option>] @ git+https://github.com/arviz-devs/arviz-base"
```
:::
::::

where `<option>` can be one of:

* `netcdf`
* `h5netcdf`
* `zarr`
* `test` (for developers)
* `doc` (for developers)


You can install multiple bundles of optional dependencies separating them with commas.
Thus, to install all user facing optional dependencies you should use `arviz-base[netcdf,zarr]`

:::{toctree}
:caption: Tutorials
:hidden:

how_to/ConversionGuideEmcee
tutorial/WorkingWithDataTree
ArviZ in Context <https://arviz-devs.github.io/Exploratory-Analysis-of-Bayesian-Models/>
:::

:::{toctree}
:caption: Reference
:hidden:

api/index
:::

:::{toctree}
:caption: About
:hidden:

BlueSky <https://bsky.app/profile/arviz.bsky.social>
Mastodon <https://bayes.club/@ArviZ>
GitHub repository <https://github.com/arviz-devs/arviz-base>
:::
