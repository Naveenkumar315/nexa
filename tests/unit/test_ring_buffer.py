import numpy as np
import pytest

from app.audio.ring_buffer import RingBuffer


def ramp(start: int, count: int) -> np.ndarray:
    return np.arange(start, start + count, dtype=np.int16)


def test_rejects_zero_capacity():
    with pytest.raises(ValueError):
        RingBuffer(0)


def test_empty_buffer_reads_nothing():
    rb = RingBuffer(100)
    assert rb.available == 0
    assert rb.read_last(10).size == 0


def test_partial_fill_returns_what_it_has():
    rb = RingBuffer(100)
    rb.write(ramp(0, 30))
    assert rb.available == 30
    assert rb.read_last(50).size == 30
    np.testing.assert_array_equal(rb.read_last(30), ramp(0, 30))


def test_read_last_returns_most_recent_oldest_first():
    rb = RingBuffer(100)
    rb.write(ramp(0, 50))
    np.testing.assert_array_equal(rb.read_last(10), ramp(40, 10))


def test_exact_fill():
    rb = RingBuffer(100)
    rb.write(ramp(0, 100))
    assert rb.available == 100
    np.testing.assert_array_equal(rb.read_last(100), ramp(0, 100))


def test_wraps_and_discards_oldest():
    rb = RingBuffer(100)
    rb.write(ramp(0, 80))
    rb.write(ramp(80, 40))  # wraps: 0-19 are overwritten
    assert rb.available == 100
    np.testing.assert_array_equal(rb.read_last(100), ramp(20, 100))


def test_read_across_the_wrap_point():
    rb = RingBuffer(100)
    rb.write(ramp(0, 95))
    rb.write(ramp(95, 10))  # write_pos now at 5
    np.testing.assert_array_equal(rb.read_last(20), ramp(85, 20))


def test_frame_larger_than_capacity_keeps_the_tail():
    rb = RingBuffer(50)
    rb.write(ramp(0, 120))
    assert rb.available == 50
    np.testing.assert_array_equal(rb.read_last(50), ramp(70, 50))


def test_many_small_writes_stay_consistent():
    rb = RingBuffer(64)
    for i in range(0, 1000, 7):
        rb.write(ramp(i, 7))
    assert rb.available == 64
    last = rb.read_last(64)
    np.testing.assert_array_equal(last, np.arange(last[0], last[0] + 64, dtype=np.int16))


def test_clear_resets():
    rb = RingBuffer(100)
    rb.write(ramp(0, 50))
    rb.clear()
    assert rb.available == 0
    assert rb.read_last(10).size == 0