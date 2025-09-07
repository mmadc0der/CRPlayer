import argparse
import asyncio
import sys

from core.stream_pipeline import SharedStreamBuffer
from .server import ReplayerServer
from .stream_adapter import StreamToWebsocketAdapter
from .producer import LiveProducer, ReplayProducer


def parse_args(argv):
  ap = argparse.ArgumentParser(description="Replayer tool: live or replay mode broadcasting over WebSocket")
  sub = ap.add_subparsers(dest="mode", required=True)

  live = sub.add_parser("live", help="Stream live from Android via tools.streamer")
  live.add_argument("--device-id", dest="device_id", default=None)
  live.add_argument("--use-gpu", dest="use_gpu", action="store_true")
  live.add_argument("--max-fps", dest="max_fps", type=int, default=60)
  live.add_argument("--max-size", dest="max_size", type=int, default=1600)
  live.add_argument("--codec", dest="codec", default="h264")
  live.add_argument("--bitrate", dest="bitrate", type=int, default=None)

  replay = sub.add_parser("replay", help="Replay from directory or numpy file")
  replay.add_argument("source", help="Directory of images or .npy/.npz file")
  replay.add_argument("--fps", dest="fps", type=int, default=60)
  replay.add_argument("--no-loop", dest="loop", action="store_false")

  ap.add_argument("--host", dest="host", default="0.0.0.0")
  ap.add_argument("--port", dest="port", type=int, default=8765)
  return ap.parse_args(argv)


async def _run_live(args):
  buffer = SharedStreamBuffer(max_buffer_size=100)
  server = ReplayerServer(host=args.host, port=args.port)
  adapter = StreamToWebsocketAdapter(buffer, server, fps=args.max_fps)
  producer = LiveProducer(buffer, device_id=args.device_id, use_gpu=args.use_gpu, max_fps=args.max_fps,
                          max_size=args.max_size, codec=args.codec, bitrate=args.bitrate)
  await server.start()
  producer.start()
  await adapter.start()
  print(f"[ReplayerCLI] Live mode started at {server.url}")
  try:
    while True:
      await asyncio.sleep(1.0)
  except (KeyboardInterrupt, asyncio.CancelledError):
    pass
  finally:
    await adapter.stop()
    producer.stop()
    await server.stop()


async def _run_replay(args):
  buffer = SharedStreamBuffer(max_buffer_size=100)
  server = ReplayerServer(host=args.host, port=args.port)
  adapter = StreamToWebsocketAdapter(buffer, server, fps=args.fps)
  producer = ReplayProducer(buffer, source_path=args.source, fps=args.fps, loop=args.loop)
  await server.start()
  producer.start()
  await adapter.start()
  print(f"[ReplayerCLI] Replay mode started at {server.url}")
  try:
    while producer.is_producing:
      await asyncio.sleep(0.5)
  except (KeyboardInterrupt, asyncio.CancelledError):
    pass
  finally:
    await adapter.stop()
    producer.stop()
    await server.stop()


def main(argv=None):
  if argv is None:
    argv = sys.argv[1:]
  args = parse_args(argv)
  if args.mode == "live":
    asyncio.run(_run_live(args))
    return 0
  elif args.mode == "replay":
    asyncio.run(_run_replay(args))
    return 0
  return 2


if __name__ == "__main__":
  raise SystemExit(main())

