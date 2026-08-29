# pylint: disable=no-self-use
"""Tests for labeller classes."""

import pytest

from arviz_base.labels import (
    BaseLabeller,
    DimCoordLabeller,
    DimIdxLabeller,
    IdxLabeller,
    MapLabeller,
    NoVarLabeller,
    mix_labellers,
)


class Data:
    def __init__(self):
        self.sel = {
            "instrument": "a",
            "experiment": 3,
        }
        self.isel = {
            "instrument": 0,
            "experiment": 4,
        }


@pytest.fixture(scope="module")
def multidim_sels():
    return Data()


def test_mix_labellers():
    sel = {"dim1": "a", "dim2": "top"}
    mix_labeller = mix_labellers((MapLabeller, DimCoordLabeller))(
        dim_map={"dim1": "$d_1$", "dim2": r"$d_2$"}
    )
    label = mix_labeller.sel_to_str(sel, sel)
    assert label == "$d_1$: a, $d_2$: top"


class TestLabellers:
    @pytest.fixture(scope="class")
    @classmethod
    def labellers(cls):
        return {
            "BaseLabeller": BaseLabeller(),
            "DimCoordLabeller": DimCoordLabeller(),
            "IdxLabeller": IdxLabeller(),
            "DimIdxLabeller": DimIdxLabeller(),
            "MapLabeller": MapLabeller(
                var_name_map={"theta": r"$\theta$"},
                dim_map={"school": "School"},
                coord_map={"instrument": {"a": "ATHENA"}},
            ),
            "NoVarLabeller": NoVarLabeller(),
        }

    # pylint: disable=redefined-outer-name
    @pytest.mark.parametrize(
        "args",
        [
            ("BaseLabeller", "theta\na, 3"),
            ("DimCoordLabeller", "theta\ninstrument: a, experiment: 3"),
            ("IdxLabeller", "theta\n0, 4"),
            ("DimIdxLabeller", "theta\ninstrument#0, experiment#4"),
            ("MapLabeller", "$\\theta$\nATHENA, 3"),
            ("NoVarLabeller", "a, 3"),
        ],
    )
    def test_make_label_vert(self, args, multidim_sels, labellers):
        name, expected_label = args
        labeller_arg = labellers[name]
        label = labeller_arg.make_label_vert("theta", multidim_sels.sel, multidim_sels.isel)
        assert label == expected_label

    @pytest.mark.parametrize(
        "args",
        [
            ("BaseLabeller", "theta[a, 3]"),
            ("DimCoordLabeller", "theta[instrument: a, experiment: 3]"),
            ("IdxLabeller", "theta[0, 4]"),
            ("DimIdxLabeller", "theta[instrument#0, experiment#4]"),
            ("MapLabeller", r"$\theta$[ATHENA, 3]"),
            ("NoVarLabeller", "a, 3"),
        ],
    )
    def test_make_label_flat(self, args, multidim_sels, labellers):
        name, expected_label = args
        labeller_arg = labellers[name]
        label = labeller_arg.make_label_flat("theta", multidim_sels.sel, multidim_sels.isel)
        assert label == expected_label

    @pytest.mark.parametrize(
        "args",
        [
            ("BaseLabeller", "school", "school"),
            ("MapLabeller", "school", "School"),
            ("MapLabeller", "chain", "chain"),
        ],
    )
    def test_dim_to_str(self, args, labellers):
        name, dim, expected_label = args

        labeller_arg = labellers[name]
        label = labeller_arg.dim_to_str(dim)

        assert label == expected_label

    def test_dim_to_str_none(self, labellers):
        assert labellers["BaseLabeller"].dim_to_str(None) is None
        assert labellers["MapLabeller"].dim_to_str(None) is None
