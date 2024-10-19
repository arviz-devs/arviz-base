import textwrap


class BaseValidator:
    wrapper = textwrap.TextWrapper(width=80)

    def __init__(self, func_name):
        self.func_name = func_name

    def filter_vars(self, value):
        acceptable_filter_vars = [None, "like", "regex"]
        if value not in acceptable_filter_vars:
            msg = "\n".join(
                self.wrapper.wrap(
                    f"Input variable 'filter_vars' from {self.func_name!r} must be one of the "
                    f"following {acceptable_filter_vars}. You gave {value}."
                )
            )
            raise ValueError(msg)
        return value

    def var_names(self, value):
        """(Data) variable names."""
        msg = "\n".join(
            self.wrapper.wrap(
                f"Input vairable 'var_names' from {self.func_name!r} must be a string or a list "
                f"of strings. You gave {value}"
            )
        )
        if not isinstance(value, str) or not all(isinstance(v, str) for v in value):
            raise TypeError(msg)

        if isinstance(value, str):
            value = [value]

        # NOTE: If the given value does not exist in the data_vars, then xarray will throw an
        #       error. arviz_base.utils also throws an error, which should be allowed to propagate
        #       up to xarray.

        return value
