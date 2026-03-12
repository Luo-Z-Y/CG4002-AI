from __future__ import annotations

"""Connect Ultra96 to EC2 over WebSocket and relay inference requests."""

import argparse
import asyncio
import contextlib

try:
    import websockets
except ImportError as exc:
    raise SystemExit("The `websockets` package is required for ultra96/ec2_bridge.py") from exc

try:
    from . import hardware as hw
    from .receiver import Ultra96Receiver
    from .runtime import Ultra96Runtime
    from .sender import Ultra96Sender
    from .service import Ultra96Service
except ImportError:
    import hardware as hw
    from receiver import Ultra96Receiver
    from runtime import Ultra96Runtime
    from sender import Ultra96Sender
    from service import Ultra96Service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultra96 bridge for EC2 WebSocket requests")
    parser.add_argument("--ws-url", required=True, help="WebSocket URL exposed by the EC2-side app")
    parser.add_argument("--xsa-path", default=hw.DEFAULT_XSA)
    parser.add_argument("--gesture-core", default=hw.GESTURE_CORE_NAME)
    parser.add_argument("--voice-core", default=hw.VOICE_CORE_NAME)
    parser.add_argument("--gesture-dma", default=hw.GESTURE_DMA_NAME)
    parser.add_argument("--voice-dma", default=hw.VOICE_DMA_NAME)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--default-confidence", type=float, default=1.0)
    parser.add_argument("--voice-labels", default=",".join(hw.VOICE_LABELS))
    parser.add_argument("--gesture-labels", default=",".join(hw.GESTURE_LABELS))
    parser.add_argument("--reconnect-delay-s", type=float, default=3.0)
    return parser.parse_args()


def _parse_labels(raw: str) -> list[str]:
    labels = [item.strip() for item in raw.split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one label is required")
    return labels


async def _run_bridge(args: argparse.Namespace) -> None:
    runtime = Ultra96Runtime(
        xsa_path=args.xsa_path,
        gesture_core_name=args.gesture_core,
        voice_core_name=args.voice_core,
        gesture_dma_name=args.gesture_dma,
        voice_dma_name=args.voice_dma,
        timeout_s=args.timeout_s,
        default_confidence=args.default_confidence,
        gesture_labels=_parse_labels(args.gesture_labels),
        voice_labels=_parse_labels(args.voice_labels),
    )
    try:
        while True:
            print(f"Connecting to EC2 websocket: {args.ws_url}")
            try:
                async with websockets.connect(args.ws_url, max_size=None) as websocket:
                    sender = Ultra96Sender(websocket)
                    receiver = Ultra96Receiver()
                    service = Ultra96Service(sender, receiver, runtime)
                    print("Connected. Waiting for EC2 requests.")
                    await service.serve_forever(websocket)
            except Exception as exc:
                print(f"[WARN] bridge disconnected: {exc}")
                await asyncio.sleep(args.reconnect_delay_s)
    finally:
        with contextlib.suppress(Exception):
            runtime.close()


def main() -> None:
    args = parse_args()
    asyncio.run(_run_bridge(args))


if __name__ == "__main__":
    main()
