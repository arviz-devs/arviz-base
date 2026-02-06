# Updating the example data

Metadata for ArviZ's example datasets (which are loadable by {func}`~arviz_base.load_arviz_data`) is stored in the [`arviz_example_data`](https://github.com/arviz-devs/arviz_example_data) repository.
This repo has been embedded into the `arviz` repo at `/src/arviz_base/example_data` using [git subtree](https://www.atlassian.com/git/tutorials/git-subtree) with the following command:

```bash
$ git subtree add --prefix src/arviz_base/example_data https://github.com/arviz-devs/arviz_example_data.git main --squash
```

When `arviz_example_data` is updated, the subtree within the `arviz` repo also needs to be updated with the following command:

```bash
$ git subtree pull --prefix src/arviz_base/example_data https://github.com/arviz-devs/arviz_example_data.git main --squash
```

Moreover, to keep the size of the wheel and sdist for `arviz-base` as small as possible, the `code` folder of `arviz_example_data` is also removed as it contains some jupyter notebooks and csv files.
