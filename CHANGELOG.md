<a id="v0.8.0"></a>
# [v0.8.0](https://github.com/arviz-devs/arviz-base/releases/tag/v0.8.0) - 2026-01-15

## What's Changed

### New Features

* Add `"round_to"` entry to `rcParams` by [@aloctavodia](https://github.com/aloctavodia) in [#120](https://github.com/arviz-devs/arviz-base/pull/120)

### Maintenance and bug fixes

* Bump the actions group with 2 updates by [@dependabot](https://github.com/dependabot)[bot] in [#125](https://github.com/arviz-devs/arviz-base/pull/125)
* Use Zenodo as a backup source for datasets by [@aloctavodia](https://github.com/aloctavodia) in [#127](https://github.com/arviz-devs/arviz-base/pull/127)

### Documentation

* Update labels related imports and cross-references by [@OriolAbril](https://github.com/OriolAbril) in [#118](https://github.com/arviz-devs/arviz-base/pull/118)
* Correct typo by [@star1327p](https://github.com/star1327p) in [#119](https://github.com/arviz-devs/arviz-base/pull/119)
* Add roaches example by [@aloctavodia](https://github.com/aloctavodia) in [#121](https://github.com/arviz-devs/arviz-base/pull/121) and [#122](https://github.com/arviz-devs/arviz-base/pull/122)

## New Contributors

* [@tomicapretto](https://github.com/tomicapretto) made their first contribution in [#129](https://github.com/arviz-devs/arviz-base/pull/129)

**Full Changelog**: https://github.com/arviz-devs/arviz-base/compare/v0.7.0...v0.8.0

[Changes][v0.8.0]


<a id="v0.7.0"></a>
# [v0.7.0](https://github.com/arviz-devs/arviz-base/releases/tag/v0.7.0) - 2025-11-11

## What's Changed

### New Features

* add some more flexibility to dataset_to conveters by [@OriolAbril](https://github.com/OriolAbril) in [#63](https://github.com/arviz-devs/arviz-base/pull/63)
* adds label_type parameter in dataset_to_dataarray function by [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) in [#81](https://github.com/arviz-devs/arviz-base/pull/81)
* Add `get_unconstrained_samples` function by [@symeneses](https://github.com/symeneses) in [#82](https://github.com/arviz-devs/arviz-base/pull/82)
* Add function to retrieve citations for arviz or methods implemented in arviz by [@aloctavodia](https://github.com/aloctavodia) in [#77](https://github.com/arviz-devs/arviz-base/pull/77) and in [#89](https://github.com/arviz-devs/arviz-base/pull/89)
* add flag to disable checking in variables are present by [@aloctavodia](https://github.com/aloctavodia) in [#93](https://github.com/arviz-devs/arviz-base/pull/93)
* Numpyro MCMC autodims migration by [@kylejcaron](https://github.com/kylejcaron) in [#88](https://github.com/arviz-devs/arviz-base/pull/88)
* Set default ci_prob to 0.89 by [@aloctavodia](https://github.com/aloctavodia) in [#99](https://github.com/arviz-devs/arviz-base/pull/99)
* Add censored cats example and test datatree for plot_pcc_censored by [@aloctavodia](https://github.com/aloctavodia) in [#100](https://github.com/arviz-devs/arviz-base/pull/100)
* Add PyStan converter by [@aloctavodia](https://github.com/aloctavodia) in [#47](https://github.com/arviz-devs/arviz-base/pull/47)


### Maintenance and bug fixes
* Revert docstub install and update version by [@aloctavodia](https://github.com/aloctavodia) in [#74](https://github.com/arviz-devs/arviz-base/pull/74)
* Type checking by [@symeneses](https://github.com/symeneses) in [#75](https://github.com/arviz-devs/arviz-base/pull/75)
* Fix CI error due to jax-numpyro incompatibility by [@OriolAbril](https://github.com/OriolAbril) in [#84](https://github.com/arviz-devs/arviz-base/pull/84)
* Bump actions/checkout from 4 to 5 by [@dependabot](https://github.com/dependabot)[bot] in [#91](https://github.com/arviz-devs/arviz-base/pull/91)
* Bump actions/download-artifact from 4 to 5 by [@dependabot](https://github.com/dependabot)[bot] in [#92](https://github.com/arviz-devs/arviz-base/pull/92)
* Patch CI so it passes again by [@OriolAbril](https://github.com/OriolAbril) in [#94](https://github.com/arviz-devs/arviz-base/pull/94)
* Bump actions/setup-python from 5 to 6 by [@dependabot](https://github.com/dependabot)[bot] in [#96](https://github.com/arviz-devs/arviz-base/pull/96)
* Update docstub and improve some type hints by [@OriolAbril](https://github.com/OriolAbril) in [#86](https://github.com/arviz-devs/arviz-base/pull/86)
* Fix and update CI by [@OriolAbril](https://github.com/OriolAbril) in [#111](https://github.com/arviz-devs/arviz-base/pull/111)
* Added `from_numpyro_svi` converter by [@kylejcaron](https://github.com/kylejcaron) in [#95](https://github.com/arviz-devs/arviz-base/pull/95)
* Add dataset for testing R2 function and add reference pseudo-variance bernoulli by [@aloctavodia](https://github.com/aloctavodia) in [#115](https://github.com/arviz-devs/arviz-base/pull/115)


### Documentation
* Add new citations from `loo_score()` by [@jordandeklerk](https://github.com/jordandeklerk) in [#98](https://github.com/arviz-devs/arviz-base/pull/98)
* Add Kaplan Meir reference by [@aloctavodia](https://github.com/aloctavodia) in [#101](https://github.com/arviz-devs/arviz-base/pull/101)
* Correct two more typos in documentation by [@star1327p](https://github.com/star1327p) in [#102](https://github.com/arviz-devs/arviz-base/pull/102)
* Fix typo compabible -> compatible by [@star1327p](https://github.com/star1327p) in [#103](https://github.com/arviz-devs/arviz-base/pull/103)
* Add cross-ref link to the function `arviz_base.clear_data_home` by [@star1327p](https://github.com/star1327p) in [#104](https://github.com/arviz-devs/arviz-base/pull/104)
* Correct a few formatting issues in Working with Data Tree by [@star1327p](https://github.com/star1327p) in [#112](https://github.com/arviz-devs/arviz-base/pull/112)
* Fix typo "he order" -> "the order" by [@star1327p](https://github.com/star1327p) in [#114](https://github.com/arviz-devs/arviz-base/pull/114)


## New Contributors
* [@symeneses](https://github.com/symeneses) made their first contribution in [#75](https://github.com/arviz-devs/arviz-base/pull/75)
* [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) made their first contribution in [#81](https://github.com/arviz-devs/arviz-base/pull/81)
* [@kylejcaron](https://github.com/kylejcaron) made their first contribution in [#88](https://github.com/arviz-devs/arviz-base/pull/88)
* [@jordandeklerk](https://github.com/jordandeklerk) made their first contribution in [#98](https://github.com/arviz-devs/arviz-base/pull/98)

**Full Changelog**: https://github.com/arviz-devs/arviz-base/compare/v0.6.0...v0.7.0

[Changes][v0.7.0]


<a id="v0.6.0"></a>
# [v0.6.0](https://github.com/arviz-devs/arviz-base/releases/tag/v0.6.0) - 2025-06-16

## What's Changed

## New Features
* Add references_to_dataset function by [@OriolAbril](https://github.com/OriolAbril) in [#50](https://github.com/arviz-devs/arviz-base/pull/50)
* Explicitly load example datasets by [@OriolAbril](https://github.com/OriolAbril) in [#51](https://github.com/arviz-devs/arviz-base/pull/51)
* Add support for nd arrays in references_to_dataset by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#53](https://github.com/arviz-devs/arviz-base/pull/53)
* Support stacked sample dims in dataset_to_dataarray by [@OriolAbril](https://github.com/OriolAbril) in [#60](https://github.com/arviz-devs/arviz-base/pull/60)

## Maintenance and bug fixes
* Improve publish workflow by [@OriolAbril](https://github.com/OriolAbril) in [#52](https://github.com/arviz-devs/arviz-base/pull/52)
* Type hints from docstrings by [@OriolAbril](https://github.com/OriolAbril) in [#54](https://github.com/arviz-devs/arviz-base/pull/54)
* Some updates to pre-commit and tox -e check by [@OriolAbril](https://github.com/OriolAbril) in [#58](https://github.com/arviz-devs/arviz-base/pull/58)
* Move some datasets from arviz-plots and arviz-stats by [@aloctavodia](https://github.com/aloctavodia) in [#65](https://github.com/arviz-devs/arviz-base/pull/65)
* add tests for automatic naming of dimensions by [@OriolAbril](https://github.com/OriolAbril) in [#29](https://github.com/arviz-devs/arviz-base/pull/29)
* Expose testing module by [@aloctavodia](https://github.com/aloctavodia) in [#68](https://github.com/arviz-devs/arviz-base/pull/68)
* clean rcparams by [@aloctavodia](https://github.com/aloctavodia) in [#69](https://github.com/arviz-devs/arviz-base/pull/69)


## Documentation
* Add new examples by [@aloctavodia](https://github.com/aloctavodia) in [#43](https://github.com/arviz-devs/arviz-base/pull/43)
* Correct a class label ref in  WorkingWithDataTree.ipynb by [@star1327p](https://github.com/star1327p) in [#45](https://github.com/arviz-devs/arviz-base/pull/45)
* Correct a typo for "data reorganization" by [@star1327p](https://github.com/star1327p) in [#48](https://github.com/arviz-devs/arviz-base/pull/48)
* Fix link for ArviZ in Context by [@star1327p](https://github.com/star1327p) in [#49](https://github.com/arviz-devs/arviz-base/pull/49)
* Correct a few typos in WorkingWithDataTree.ipynb by [@star1327p](https://github.com/star1327p) in [#57](https://github.com/arviz-devs/arviz-base/pull/57)
* Improve docstring for convert_to_dataset by [@Quantum-Kayak](https://github.com/Quantum-Kayak) in [#56](https://github.com/arviz-devs/arviz-base/pull/56)
* Correct a method label ref in Conversion Guide Emcee by [@star1327p](https://github.com/star1327p) in [#62](https://github.com/arviz-devs/arviz-base/pull/62)

## New Contributors
* [@star1327p](https://github.com/star1327p) made their first contribution in [#45](https://github.com/arviz-devs/arviz-base/pull/45)
* [@rohanbabbar04](https://github.com/rohanbabbar04) made their first contribution in [#53](https://github.com/arviz-devs/arviz-base/pull/53)
* [@Quantum-Kayak](https://github.com/Quantum-Kayak) made their first contribution in [#56](https://github.com/arviz-devs/arviz-base/pull/56)

**Full Changelog**: https://github.com/arviz-devs/arviz-base/compare/v0.5.0...v0.6.0

[Changes][v0.6.0]


<a id="v0.5.0"></a>
# [v0.5.0](https://github.com/arviz-devs/arviz-base/releases/tag/v0.5.0) - 2025-03-20

## What's Changed
* Change default ci_prob to 0.94 by [@aloctavodia](https://github.com/aloctavodia) in [#37](https://github.com/arviz-devs/arviz-base/pull/37)
* Add SBC datatree example by [@aloctavodia](https://github.com/aloctavodia) in [#38](https://github.com/arviz-devs/arviz-base/pull/38)
* Add `from_numpyro` converter by [@aloctavodia](https://github.com/aloctavodia) in [#39](https://github.com/arviz-devs/arviz-base/pull/39)


## New Contributors
* [@github-actions](https://github.com/github-actions) made their first contribution in [#35](https://github.com/arviz-devs/arviz-base/pull/35)

**Full Changelog**: https://github.com/arviz-devs/arviz-base/compare/v0.4.0...v0.5.0

[Changes][v0.5.0]


<a id="v0.4.0"></a>
# [v0.4.0](https://github.com/arviz-devs/arviz-base/releases/tag/v0.4.0) - 2025-03-05

## What's Changed
* post release tasks and update ci versions by [@OriolAbril](https://github.com/OriolAbril) in [#27](https://github.com/arviz-devs/arviz-base/pull/27)
* use DataTree from xarray instead of from xarray-datatree by [@OriolAbril](https://github.com/OriolAbril) in [#24](https://github.com/arviz-devs/arviz-base/pull/24)
* add dataset->stacked dataarray/dataframe converters by [@OriolAbril](https://github.com/OriolAbril) in [#25](https://github.com/arviz-devs/arviz-base/pull/25)
* keepdataset by [@aloctavodia](https://github.com/aloctavodia) in [#30](https://github.com/arviz-devs/arviz-base/pull/30)
* Add crabs datasets by [@aloctavodia](https://github.com/aloctavodia) in [#31](https://github.com/arviz-devs/arviz-base/pull/31)
* Automatic changelog by [@aloctavodia](https://github.com/aloctavodia) in [#32](https://github.com/arviz-devs/arviz-base/pull/32)



**Full Changelog**: https://github.com/arviz-devs/arviz-base/compare/v0.3.0...v0.4.0

[Changes][v0.4.0]


[v0.8.0]: https://github.com/arviz-devs/arviz-base/compare/v0.7.0...v0.8.0
[v0.7.0]: https://github.com/arviz-devs/arviz-base/compare/v0.6.0...v0.7.0
[v0.6.0]: https://github.com/arviz-devs/arviz-base/compare/v0.5.0...v0.6.0
[v0.5.0]: https://github.com/arviz-devs/arviz-base/compare/v0.4.0...v0.5.0
[v0.4.0]: https://github.com/arviz-devs/arviz-base/tree/v0.4.0

<!-- Generated by https://github.com/rhysd/changelog-from-release v3.9.1 -->
