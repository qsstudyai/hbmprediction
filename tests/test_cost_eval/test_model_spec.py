import unittest

from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.layers import build_dense_decoder, build_moe_decoder


def dense_dims(**overrides):
    values = dict(H=512, F=1024, n_heads=8, n_kv=8, head_dim=64, S=256, B=1, vocab=32000, n_layers=2)
    values.update(overrides)
    return DimTable(**values)


class ModelSpecTests(unittest.TestCase):
    def test_model_spec_requires_one_pattern_entry_per_layer(self):
        dims = dense_dims()
        with self.assertRaisesRegex(ValueError, "layer_pattern"):
            ModelSpec("broken", dims, ("dense",), {"dense": build_dense_decoder(dims)})

    def test_builtin_dense_and_moe_graphs_have_memory_contracts(self):
        dense = build_dense_decoder(dense_dims())
        self.assertTrue(any(op.params for op in dense.ops))
        self.assertTrue(any(op.saves for op in dense.ops))
        moe_dims = dense_dims(n_experts=8, topk=2, moe_F=1024)
        moe = build_moe_decoder(moe_dims)
        self.assertTrue(any(tensor.is_expert for op in moe.ops for tensor in op.params))
