import csv
import json
import tempfile
import unittest
from pathlib import Path

from cost_eval import (
    DimTable, Evaluator, HardwareSpec, ModelSpec, OptimizerSpec,
    ParallelConfig,
)
from cost_eval.layers import build_dense_decoder


class MemoryTraceTests(unittest.TestCase):
    def _evaluate(self, capture=True):
        dims = DimTable(
            H=16, F=32, n_heads=4, n_kv=2, head_dim=4,
            S=16, B=1, vocab=32, n_layers=1,
        )
        spec = ModelSpec(
            "trace-dense", dims, ("dense",),
            {"dense": build_dense_decoder(dims)},
        )
        evaluator = Evaluator(
            spec, ParallelConfig(), OptimizerSpec.adamw(),
            HardwareSpec(10**12, allocator_granularity_bytes=1),
        )
        return evaluator.evaluate_with_trace(capture)

    def test_trace_reconciles_peak_and_exposes_every_operator(self):
        report, trace = self._evaluate()
        stage = report.per_stage[0]
        raw_peak = max(
            event.total_active_bytes
            for event in trace.events if event.stage == 0
        )
        self.assertEqual(raw_peak, stage.physical_dynamic_peak_bytes)
        self.assertEqual(
            trace.stage_compositions[0]["total_point_bytes"],
            stage.total_point_bytes,
        )

        forward_markers = {
            event.op_name for event in trace.events
            if event.phase == "fwd" and event.action_kind == "MARK"
            and event.layer_id == 0
        }
        self.assertIn("rope", forward_markers)
        self.assertIn("add2", forward_markers)
        backward_ops = {
            event.op_name for event in trace.events
            if event.phase == "bwd" and event.tag.startswith("bwd_op@0:")
        }
        self.assertIn("rope", backward_ops)
        self.assertIn("add2", backward_ops)
        self.assertTrue(any(
            event.action_kind == "ALLOC" and event.tensor_key.startswith("qkv|")
            for event in trace.events
        ))
        self.assertTrue(any(
            event.tag == "bwd_activation_release@0"
            and event.bucket_deltas["act_live"] < 0
            for event in trace.events
        ))
        self.assertTrue(any(event.live_tensors for event in trace.events))

    def test_json_and_csv_exports_are_machine_readable(self):
        _, trace = self._evaluate(capture=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = trace.write(root / "trace.json")
            csv_path = trace.write(root / "trace.csv")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], trace.schema_version)
            self.assertEqual(len(payload["events"]), len(trace.events))
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(trace.events))
            self.assertIn("delta_act_live", rows[0])


if __name__ == "__main__":
    unittest.main()
