"""ArviZ testing utilities."""

from datatree import DataTree


def check_multiple_attrs(
    test_dict: dict[str, list[str]], parent: DataTree
) -> list[str | tuple[str, str]]:
    """Perform multiple hasattr checks on InferenceData objects.

    It is thought to first check if the parent object contains a given dataset,
    and then (if present) check the attributes of the dataset.

    Given the output of the function, all mismatches between expectation and reality can
    be retrieved: a single string indicates a group mismatch and a tuple of strings
    ``(group, var)`` indicates a mismatch in the variable ``var`` of ``group``.

    Parameters
    ----------
    test_dict: dict of {str : list of str}
        Its structure should be `{dataset1_name: [var1, var2], dataset2_name: [var]}`.
        A ``~`` at the beginning of a dataset or variable name indicates the name NOT
        being present must be asserted.
    parent: InferenceData
        InferenceData object on which to check the attributes.

    Returns
    -------
    list
        List containing the failed checks. It will contain either the dataset_name or a
        tuple (dataset_name, var) for all non present attributes.

    Examples
    --------
    The output below indicates that ``posterior`` group was expected but not found, and
    variables ``a`` and ``b``:

        ["posterior", ("prior", "a"), ("prior", "b")]

    Another example could be the following:

        [("posterior", "a"), "~observed_data", ("sample_stats", "~log_likelihood")]

    In this case, the output indicates that variable ``a`` was not found in ``posterior``
    as it was expected, however, in the other two cases, the preceding ``~`` (kept from the
    input negation notation) indicates that ``observed_data`` group should not be present
    but was found in the InferenceData and that ``log_likelihood`` variable was found
    in ``sample_stats``, also against what was expected.

    """
    failed_attrs: list[str | tuple[str, str]] = []
    for dataset_name, attributes in test_dict.items():
        if dataset_name.startswith("~"):
            if hasattr(parent, dataset_name[1:]):
                failed_attrs.append(dataset_name)
        elif hasattr(parent, dataset_name):
            dataset = getattr(parent, dataset_name)
            for attribute in attributes:
                if attribute.startswith("~"):
                    if hasattr(dataset, attribute[1:]):
                        failed_attrs.append((dataset_name, attribute))
                elif not hasattr(dataset, attribute):
                    failed_attrs.append((dataset_name, attribute))
        else:
            failed_attrs.append(dataset_name)
    return failed_attrs
