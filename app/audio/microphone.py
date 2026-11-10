import queue
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

log = get_logger(__name__)


class Microphone:
    """Captures int16 mono frames and pushes them onto a queue.

    The callback runs on PortAudio's thread. It must not block.
    """

    def __init__(
        self,
        device: int | None = None,
        samplerate: int = 16_000,
        channels: int = 1,
        frame_size: int = 1280,
        max_queue: int = 200,
    ) -> None:
        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self.frame_size = frame_size
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue)
        self._stream: sd.InputStream | None = None
        self.dropped = 0

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.warning("audio.stream.status", status=str(status))
        try:
            # .copy() is mandatory — PortAudio reuses this buffer.
            self._queue.put_nowait(indata.copy().reshape(-1))
        except queue.Full:
            self.dropped += 1

    def start(self) -> None:
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.frame_size,
            callback=self._callback,
        )
        self._stream.start()
        log.info(
            "audio.stream.started",
            device=self.device,
            samplerate=self.samplerate,
            frame_size=self.frame_size,
        )

    def frames(self, timeout: float = 1.0):
        """Yield frames as they arrive. Stops if nothing comes for `timeout`."""
        while True:
            try:
                yield self._queue.get(timeout=timeout)
            except queue.Empty:
                log.warning("audio.stream.starved")
                return

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        log.info("audio.stream.stopped", dropped=self.dropped)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def rms_dbfs(frame: np.ndarray) -> float:
    """Loudness of a frame, in decibels relative to full scale."""
    rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
    if rms < 1.0:
        return -96.0
    return 20.0 * np.log10(rms / 32768.0)


def main(seconds: float = 5.0) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    target_frames = int(seconds * settings.sample_rate / settings.frame_size)
    collected: list[np.ndarray] = []

    print(f"\nRecording {seconds:.0f}s on device {settings.input_device_index}. Say something.\n")

    with Microphone(
        device=settings.input_device_index,
        samplerate=settings.sample_rate,
        channels=settings.channels,
        frame_size=settings.frame_size,
    ) as mic:
        for index, frame in enumerate(mic.frames()):
            collected.append(frame)
            db = rms_dbfs(frame)
            bars = int(max(0.0, min(1.0, (db + 60.0) / 60.0)) * 40)
            print(f"\r{db:6.1f} dBFS |{'#' * bars:<40}|", end="", flush=True)
            if index + 1 >= target_frames:
                break

    audio = np.concatenate(collected)
    out_path = Path("data/recordings/test.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(out_path), "wb") as wav:
        wav.setnchannels(settings.channels)
        wav.setsampwidth(2)  # int16
        wav.setframerate(settings.sample_rate)
        wav.writeframes(audio.tobytes())

    print(f"\n\nFrames: {len(collected)}  Samples: {audio.size}  "
          f"Duration: {audio.size / settings.sample_rate:.2f}s")
    print(f"Dropped: {mic.dropped}")
    print(f"Saved:   {out_path.resolve()}\n")


if __name__ == "__main__":
    main()