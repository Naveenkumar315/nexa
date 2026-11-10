import threading

import numpy as np


class RingBuffer:
    """Fixed-size circular buffer of int16 audio samples.

    Writes overwrite the oldest data. Thread-safe: the audio callback
    writes, the main thread reads.
    """

    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self.capacity = capacity_samples
        self._buffer = np.zeros(capacity_samples, dtype=np.int16)
        self._write_pos = 0
        self._filled = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> int:
        """How many samples are currently readable."""
        with self._lock:
            return self._filled

    def write(self, frame: np.ndarray) -> None:
        """Append a frame, overwriting the oldest samples if needed."""
        data = np.asarray(frame, dtype=np.int16).reshape(-1)
        n = data.size

        if n >= self.capacity:
            # Frame bigger than the buffer: keep only its tail.
            data = data[-self.capacity:]
            n = self.capacity

        with self._lock:
            end = self._write_pos + n
            if end <= self.capacity:
                self._buffer[self._write_pos:end] = data
            else:
                split = self.capacity - self._write_pos
                self._buffer[self._write_pos:] = data[:split]
                self._buffer[: n - split] = data[split:]

            self._write_pos = end % self.capacity
            self._filled = min(self._filled + n, self.capacity)

    def read_last(self, n_samples: int) -> np.ndarray:
        """Return the most recent n_samples, oldest first.

        Returns fewer if the buffer has not filled yet.
        """
        if n_samples <= 0:
            return np.zeros(0, dtype=np.int16)

        with self._lock:
            n = min(n_samples, self._filled)
            if n == 0:
                return np.zeros(0, dtype=np.int16)

            start = (self._write_pos - n) % self.capacity
            end = start + n
            if end <= self.capacity:
                return self._buffer[start:end].copy()
            split = self.capacity - start
            out = np.empty(n, dtype=np.int16)
            out[:split] = self._buffer[start:]
            out[split:] = self._buffer[: n - split]
            return out

    def clear(self) -> None:
        with self._lock:
            self._buffer.fill(0)
            self._write_pos = 0
            self._filled = 0