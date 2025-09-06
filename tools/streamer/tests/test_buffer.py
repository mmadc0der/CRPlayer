from __future__ import annotations

from streamer.buffer import FrameBuffer


def test_frame_buffer_drop_oldest():
    buf = FrameBuffer(maxlen=2, drop_policy="drop_oldest")
    f1 = (object(), 1, 1.0)
    f2 = (object(), 1, 1.0)
    f3 = (object(), 1, 1.0)
    buf.push(f1)
    buf.push(f2)
    buf.push(f3)
    latest = buf.get_latest()
    assert latest is not None
    assert latest[0] is f3
    assert buf.get_stats()["queue_size"] == 2.0


def test_frame_buffer_drop_newest():
    buf = FrameBuffer(maxlen=2, drop_policy="drop_newest")
    f1 = (object(), 1, 1.0)
    f2 = (object(), 1, 1.0)
    f3 = (object(), 1, 1.0)
    buf.push(f1)
    buf.push(f2)
    buf.push(f3)
    latest = buf.get_latest()
    assert latest is not None
    assert latest[0] is f2
    assert buf.get_stats()["queue_size"] == 2.0

