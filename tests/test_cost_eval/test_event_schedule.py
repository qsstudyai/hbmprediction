from cost_eval.event_schedule import build_1f1b, build_interleaved_1f1b


def test_interleave_one_is_exact_1f1b_compatibility():
    assert build_interleaved_1f1b(0, 2, 4, 1) == build_1f1b(0, 2, 4)


def test_interleave_forward_and_backward_chunk_order():
    events = build_interleaved_1f1b(0, 2, 2, 2)
    for kind in ("FWD", "BWD"):
        chunks = [event.chunk for event in events if event.kind == kind]
        expected_pair = [0, 1] if kind == "FWD" else [1, 0]
        assert all(chunks[i:i + 2] == expected_pair for i in range(0, len(chunks), 2))
