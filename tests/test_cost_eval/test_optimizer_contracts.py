import pytest

from cost_eval.optimizers import optimizer_step_workspace_bytes, state_bytes
from cost_eval.shape_eval import Placement, ResolvedTensor
from cost_eval.specs import OptimizerSpec


def _weight(numel=100):
    return ResolvedTensor(
        "w", (numel,), (numel,), numel, 2, True,
        placement=Placement(), value_id="w",
    )


def test_adamw_state_contract_is_component_exact():
    state = state_bytes(_weight(), OptimizerSpec.adamw(fp32_grad=True))
    assert state.parameter == 200
    assert state.gradient == 400
    assert state.master_weight == 400
    assert state.optimizer_state == 800


def test_muon_fraction_splits_muon_and_adam_groups():
    optimizer = OptimizerSpec(
        "Muon", 14, parameter_bytes=2, gradient_bytes=4,
        master_weight_bytes=4, optimizer_state_bytes=4,
        muon_ns_steps=5, muon_parameter_fraction=0.5,
        muon_momentum_bytes=4, muon_main_parameter_bytes=4,
    )
    state = state_bytes(_weight(), optimizer)
    assert state.master_weight == 400
    assert state.optimizer_state == 50 * 4 + 50 * 8
    assert optimizer_step_workspace_bytes(optimizer, 1000) == 3000


def test_unknown_optimizer_step_contract_is_rejected():
    optimizer = OptimizerSpec("Mystery", 16)
    with pytest.raises(ValueError, match="unsupported optimizer"):
        optimizer_step_workspace_bytes(optimizer, 100)
