//! Lock-free SPSC ring buffer for audio sample transfer between threads.

use rtrb::{Consumer, Producer, RingBuffer};

/// A pair of lock-free ring buffers for bidirectional audio transfer.
pub struct AudioRingBuffer {
    /// Write end for captured audio (capture thread → inference thread).
    pub capture_producer: Producer<f32>,
    /// Read end for captured audio (inference thread reads).
    pub capture_consumer: Consumer<f32>,
    /// Write end for playback audio (inference thread writes).
    pub playback_producer: Producer<f32>,
    /// Read end for playback audio (playback thread reads).
    pub playback_consumer: Consumer<f32>,
}

impl AudioRingBuffer {
    /// Create a new pair with the given capacity (in samples per channel).
    pub fn new(capacity: usize) -> Self {
        let (capture_producer, capture_consumer) = RingBuffer::new(capacity);
        let (playback_producer, playback_consumer) = RingBuffer::new(capacity);
        Self {
            capture_producer,
            capture_consumer,
            playback_producer,
            playback_consumer,
        }
    }
}
