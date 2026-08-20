from cost_eval.event_schedule import build_interleaved_1f1b


def test_interleaved_schedule_keeps_chunk_local_order():
    events = build_interleaved_1f1b(stage=0, pp=2, m=3, interleave=2)
    forwards = [(item.mb, item.chunk) for item in events if item.kind == "FWD"]
    backwards = [(item.mb, item.chunk) for item in events if item.kind == "BWD"]
    assert forwards == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert backwards == [(0, 1), (0, 0), (1, 1), (1, 0), (2, 1), (2, 0)]
