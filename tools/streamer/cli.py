from __future__ import annotations

import argparse
import json
import signal
import sys
import time

from .streamer import AndroidStreamer, StreamerConfig


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Android screen streamer (scrcpy transport)")
    ap.add_argument("action", choices=["start", "stats", "dump"], help="action to perform")
    ap.add_argument("--device-id", dest="device_id", default=None)
    ap.add_argument("--host", dest="host", default="127.0.0.1")
    ap.add_argument("--port", dest="port", type=int, default=27183)
    ap.add_argument("--adb", dest="adb_path", default="adb")
    ap.add_argument("--max-fps", dest="max_fps", type=int, default=None)
    ap.add_argument("--max-size", dest="max_size", type=int, default=None)
    ap.add_argument("--codec", dest="codec", default="h264")
    ap.add_argument("--bitrate", dest="bitrate", type=int, default=None)
    ap.add_argument("--use-gpu", dest="use_gpu", action="store_true")
    ap.add_argument("--buffer-size", dest="buffer_size", type=int, default=8)
    ap.add_argument("--drop-policy", dest="drop_policy", default="drop_oldest")
    ap.add_argument("--debug", dest="debug", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    ns = parse_args(argv)
    config = StreamerConfig(
        device_id=ns.device_id,
        socket_host=ns.host,
        socket_port=ns.port,
        adb_path=ns.adb_path,
        max_fps=ns.max_fps,
        max_size=ns.max_size,
        codec=ns.codec,
        bitrate=ns.bitrate,
        use_gpu=ns.use_gpu,
        buffer_size=ns.buffer_size,
        drop_policy=ns.drop_policy,
        debug=ns.debug,
    )
    streamer = AndroidStreamer(config)

    if ns.action == "start":
        stop = False

        def _sigint(_sig, _frm):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, _sigint)
        signal.signal(signal.SIGTERM, _sigint)
        streamer.start()
        print("stream started; press Ctrl+C to stop")
        while not stop:
            time.sleep(0.5)
            stats = streamer.get_stats()
            sys.stdout.write("\r" + json.dumps(stats))
            sys.stdout.flush()
        print()
        streamer.stop()
        return 0

    elif ns.action == "stats":
        # One-shot run for a few seconds to show fps
        streamer.start()
        time.sleep(3.0)
        stats = streamer.get_stats()
        streamer.stop()
        print(json.dumps(stats, indent=2))
        return 0

    elif ns.action == "dump":
        # Start and dump N frames
        frames_to_dump = 10
        dumped = 0

        def on_frame(_f):
            nonlocal dumped
            dumped += 1
            if dumped >= frames_to_dump:
                streamer.stop()

        streamer.start(frame_callback=on_frame)
        while dumped < frames_to_dump:
            time.sleep(0.1)
        return 0

    else:
        print("unknown action", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

