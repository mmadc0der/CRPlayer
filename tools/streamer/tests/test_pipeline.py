from __future__ import annotations

import socket
import threading
import time

import pytest

from streamer.buffer import FrameBuffer
from streamer.pipeline import BitstreamPipeline, START3, START4


class DummyDecoder:

  def __init__(self):
    self.calls = 0

  def decode_nal(self, _nal: bytes):
    self.calls += 1
    # return one fake frame tuple
    return [(object(), 1, 1.0)]


def make_connected_sockets():
  s1, s2 = socket.socketpair()
  for s in (s1, s2):
    s.setblocking(False)
  return s1, s2


def test_pipeline_parses_annex_b_and_pushes_frames():
  decoder = DummyDecoder()
  buf = FrameBuffer(maxlen=4)
  pipeline = BitstreamPipeline(decoder=decoder, frame_buffer=buf)

  s_server, s_client = make_connected_sockets()
  running = threading.Event()
  running.set()

  # Start pipeline in thread
  th = threading.Thread(target=pipeline.run, kwargs={"socket_obj": s_client, "running_event": running})
  th.start()

  # Send two NAL units with start codes
  payload = START4 + b"\x65" + b"A" * 10 + START3 + b"\x41" + b"B" * 8
  total = 0
  while total < len(payload):
    try:
      sent = s_server.send(payload[total:])
      total += sent
    except BlockingIOError:
      time.sleep(0.01)

  # Allow some processing
  time.sleep(0.2)
  running.clear()
  th.join(timeout=2.0)

  stats = buf.get_stats()
  assert stats["queue_size"] >= 1
  assert decoder.calls >= 2
