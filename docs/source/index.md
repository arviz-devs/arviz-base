# arviz-base
ArviZ base features and converters.

:::{important}
This package is still in a very early development stage.
Testing and feedback are extremely welcome, but for
general applications, the main
[ArviZ Python library](https://python.arviz.org)
should be used.
:::

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
xarray (and xarray-datatree which will at some point be merged into xarray),
NumPy and typing-extensions.

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
:hidden:

tutorial/WorkingWithDataTree
:::

:::{toctree}
:hidden:

how_to/ConversionGuideEmcee
:::

:::{toctree}
:hidden:

api/index
:::

:::{toctree}
:caption: About
:hidden:

Twitter <https://twitter.com/arviz_devs>
Mastodon <https://bayes.club/@ArviZ>
GitHub repository <https://github.com/arviz-devs/arviz-base>
:::
