"""Methods and decorators to validate ArviZ methods."""

import inspect
import itertools
from functools import wraps
from typing import Hashable, Iterable

from arviz_base.rcparams import rcParams
from arviz_base.validate.base import BaseValidator


class Extract(BaseValidator):
    def __init__(self, func_name):
        self.func_name = func_name

    def group(self, value, dt):
        """DataTree group name."""
        groups = [g for g in map(lambda g: g.replace("/", ""), dt.groups) if g]

        if not isinstance(value, str):
            raise TypeError(f"Input variable 'group' of {self.func_name!r} must be a string.")

        if value not in groups:
            raise TypeError(
                f"Input variable 'group' of {self.func_name!r} must be the name of a group in the "
                f"DataTree. Available groups are: {list(groups)}"
            )
        return value

    def sample_dims(self, value, dt, group):
        """List of dimensions considered as sampling dimensions."""
        groups = [g for g in map(lambda g: g.replace("/", ""), dt.groups) if g]
        dims = {g: list(getattr(dt, g).dims) for g in groups}

        if value is None:
            value = rcParams["data.sample_dims"]

        if not isinstance(value, list):
            value = [value]

        msg = "\n".join(
            self.wrapper.wrap(
                f"Input variable 'sample_dims' from function {self.func_name!r} must be the names "
                f"of dimensions available to the given group '{group}'. Available dimensions for "
                f"it are {dims[group]}. You gave {value}."
            )
        )
        if not isinstance(value, Iterable):
            raise TypeError(msg)

        if not all(isinstance(v, Hashable) for v in value):
            raise TypeError(msg)

        exploded_value = value
        if all(isinstance(v, tuple) for v in value):
            exploded_value = list(itertools.chain.from_iterable(value))
        if isinstance(value, list):
            if not all(isinstance(v, str) for v in exploded_value):
                raise TypeError(msg)

        if not set(exploded_value).issubset(set(dims[group])):
            msg = (
                "The DataTree does not contain the given sample_dims for the given group "
                f"'{group}'. Available sample_dims are {dims[group]}"
            )
            raise ValueError(msg)

        return value

    def resampling_method(self, value):
        """Method to use for resampling."""
        if value is None:
            value = "multinomial"

        if not isinstance(value, str):
            msg = "\n".join(
                self.wrapper.wrap(
                    f"Input variable 'resampling_method' from {self.func_name!r} must be a "
                    f"string. You gave {value}."
                )
            )
            raise TypeError(msg)

        if value not in ["multinomial", "stratified"]:
            msg = "\n".join(
                self.wrapper.wrap(
                    f"Input variable 'resampling_method' from {self.func_name!r} must be one of "
                    f"either 'multinomial' or 'stratified'. You gave {value}."
                )
            )
            raise ValueError(msg)


def extract_inputs(func):
    validator = Extract(func_name=func.__name__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        parameters = {
            k: v.default
            for k, v in inspect.signature(func).parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        parameters.update(kwargs)

        # Validate group input
        dt = parameters["data"]
        group = parameters["group"]
        kwargs["group"] = validator.group(group, dt)

        # Validate sample_dims input
        kwargs["sample_dims"] = validator.sample_dims(
            value=parameters["sample_dims"],
            dt=dt,
            group=group,
        )

        # Validate filter_vars input
        kwargs["filter_vars"] = validator.filter_vars(value=parameters["filter_vars"])

        # WARNING: At this point, sample_dims is a list, and that list can contain a single tuple
        #          of values. The switch here may need to be changed based on the length of the
        #          list, or the length of the tuple in the list.

        # Switch combined based on the length of the sample_dims input
        if len(kwargs["sample_dims"]) == 1:
            kwargs["combined"] = True

        # Handle num_samples input
        if parameters["num_samples"] is not None and not parameters["combined"]:
            raise ValueError(
                "num_samples is only compatible with combined=True or length 1 sample_dims"
            )

        # Handle weights input
        if parameters["weights"] is not None and parameters["num_samples"] is None:
            raise ValueError("weights are only compatible with num_samples")

        # Validate var_names input
        kwargs["var_names"] = validator.var_names(parameters["var_names"])

        # Validate resampling_method input
        kwargs["resampling_method"] = validator.resampling_method(
            parameters["resampling_method"],
        )

        return func(*args, **kwargs)

    return wrapper
