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


[v0.6.0]: https://github.com/arviz-devs/arviz-base/compare/v0.5.0...v0.6.0
[v0.5.0]: https://github.com/arviz-devs/arviz-base/compare/v0.4.0...v0.5.0
[v0.4.0]: https://github.com/arviz-devs/arviz-base/tree/v0.4.0

<!-- Generated by https://github.com/rhysd/changelog-from-release v3.9.0 -->
