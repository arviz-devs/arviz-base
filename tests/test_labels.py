# pylint: disable=no-self-use
"""Tests for labeller classes."""

import pytest
from arviz_base.labels import (
    BaseLabeller,
    DimCoordLabeller,
    DimIdxLabeller,
    IdxLabeller,
    MapLabeller,
    NoModelLabeller,
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
    def labellers(self):
        return {
            "BaseLabeller": BaseLabeller(),
            "DimCoordLabeller": DimCoordLabeller(),
            "IdxLabeller": IdxLabeller(),
            "DimIdxLabeller": DimIdxLabeller(),
            "MapLabeller": MapLabeller(
                var_name_map={"theta": r"$\theta$"}, coord_map={"instrument": {"a": "ATHENA"}}
            ),
            "NoVarLabeller": NoVarLabeller(),
            "NoModelLabeller": NoModelLabeller(),
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
            ("NoModelLabeller", "theta\na, 3"),
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
            ("NoModelLabeller", "theta[a, 3]"),
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
            ("BaseLabeller", "ab initio: var_sel_label"),
            ("NoModelLabeller", "var_sel_label"),
        ],
    )
    def test_make_model_label(self, args, labellers):
        name, expected_label = args
        labeller_arg = labellers[name]
        label = labeller_arg.make_model_label("ab initio", "var_sel_label")
        assert label == expected_label
