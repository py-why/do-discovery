import numpy as np
import pytest
from pywhy_graphs import StructuralCausalModel

from dodiscover.ci import FisherZCITest, GSquareCITest, KernelCITest, Oracle

seed = 12345
rng = np.random.RandomState(seed=seed)
func_uz = lambda: rng.negative_binomial(n=1, p=0.25)
func_uxy = lambda: rng.binomial(n=1, p=0.4)
func_x = lambda u_xy: 2 * u_xy
func_y = lambda x, u_xy, z: x + u_xy + z
func_z = lambda u_z: u_z

# construct the SCM and the corresponding causal graph
scm = StructuralCausalModel(
    exogenous={
        "u_xy": func_uxy,
        "u_z": func_uz,
    },
    endogenous={"x": func_x, "y": func_y, "z": func_z},
)

sample_df = scm.sample(n=100)
ground_truth_graph = scm.get_causal_graph()


@pytest.mark.parametrize(
    "ci_estimator",
    [
        KernelCITest(),
        GSquareCITest(),
        FisherZCITest(),
        Oracle(ground_truth_graph),
    ],
)
def test_ci_tests(ci_estimator):
    x = "x"
    y = "y"
    with pytest.raises(ValueError, match="The z conditioning set variables are not all"):
        ci_estimator.test(sample_df, {x}, {y}, z_covariates=["blah"])

    with pytest.raises(ValueError, match="The x variables are not all"):
        ci_estimator.test(sample_df, {"blah"}, y_vars={y}, z_covariates=["z"])

    with pytest.raises(ValueError, match="The y variables are not all"):
        ci_estimator.test(sample_df, {x}, y_vars={"blah"}, z_covariates=["z"])
