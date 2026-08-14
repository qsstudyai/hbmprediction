import unittest

from cost_eval.layers import build_dense_decoder
from cost_eval.model_spec import DimTable, ModelSpec, TensorRef
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval, eval_expr, resolve_tensor
from cost_eval.specs import ParallelConfig


def make_dims():
    return DimTable(H=512, F=1024, n_heads=8, n_kv=8, head_dim=64, S=256, B=2, vocab=32000, n_layers=6)


class ParallelAndShapeTests(unittest.TestCase):
    def test_restricted_shape_expression(self):
        self.assertEqual(eval_expr("2*F", {"F": 1024}), 2048)
        with self.assertRaises(ValueError):
            eval_expr("__import__('os')", {})

    def test_pipeline_layers_are_balanced(self):
        parallel = ParallelModel.build(ParallelConfig(pp=2), make_dims())
        self.assertEqual(parallel.stage_layers, {0: (0, 1, 2), 1: (3, 4, 5)})

    def test_tensor_sharding_resolves_local_shape(self):
        parallel = ParallelModel.build(ParallelConfig(tp=4, cp=2), make_dims())
        tensor = TensorRef("x", ("S", "B", "H"), {0: "cp", 2: "tp"})
        resolved = resolve_tensor(tensor, make_dims().as_dict(), parallel)
        self.assertEqual(resolved.global_shape, (256, 2, 512))
        self.assertEqual(resolved.local_shape, (128, 2, 128))

    def test_shape_eval_builds_every_pipeline_stage(self):
        dims = make_dims()
        spec = ModelSpec("dense", dims, ("dense",) * dims.n_layers, {"dense": build_dense_decoder(dims)})
        parallel = ParallelModel.build(ParallelConfig(tp=2, pp=2), dims)
        graph = ShapeEval().resolve(spec, parallel)
        self.assertEqual(tuple(graph.stages), (0, 1))
        self.assertEqual(sum(len(layers) for layers in graph.stages.values()), dims.n_layers)
