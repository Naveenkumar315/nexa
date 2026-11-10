from dataclasses import dataclass

import sounddevice as sd

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float
    hostapi_name: str


def list_input_devices() -> list[AudioDevice]:
    """Every device on this machine that can record."""
    hostapis = sd.query_hostapis()
    found: list[AudioDevice] = []

    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] < 1:
            continue
        found.append(
            AudioDevice(
                index=index,
                name=device["name"],
                max_input_channels=device["max_input_channels"],
                default_samplerate=device["default_samplerate"],
                hostapi_name=hostapis[device["hostapi"]]["name"],
            )
        )
    return found


def supports(device_index: int, samplerate: int, channels: int) -> bool:
    """Can this device actually give us the format we need?"""
    try:
        sd.check_input_settings(
            device=device_index,
            samplerate=samplerate,
            channels=channels,
            dtype="int16",
        )
        return True
    except Exception as exc:
        log.debug("device.unsupported", index=device_index, error=str(exc))
        return False


def default_input_index() -> int | None:
    try:
        index = sd.default.device[0]
    except Exception:
        return None
    return index if isinstance(index, int) and index >= 0 else None


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    devices = list_input_devices()
    if not devices:
        print("No input devices found. Check Windows sound settings.")
        return

    default = default_input_index()
    print(f"\nTarget format: {settings.sample_rate} Hz, {settings.channels} ch, int16\n")

    for device in devices:
        ok = supports(device.index, settings.sample_rate, settings.channels)
        mark = "OK" if ok else "--"
        star = "  <-- Windows default" if device.index == default else ""
        print(
            f"[{mark}] {device.index:>3}  {device.name[:45]:<45} "
            f"{device.hostapi_name:<22} {device.max_input_channels}ch "
            f"{device.default_samplerate:>6.0f} Hz{star}"
        )
    print()


if __name__ == "__main__":
    main()