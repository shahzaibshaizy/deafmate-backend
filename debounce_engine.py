"""
Temporal Debouncing and Gesture Stabilization Engine.

This module solves the "flicker" and "character spam" problems in real-time
computer vision sign language applications.

Key Problems Solved:
1. Frame Jitter: Raw single-frame inferences fluctuate due to camera noise.
2. In-Between Transitions: Moving fingers from sign 'A' to sign 'B' produces
   spurious intermediate guesses.
3. Duplication Spam: Holding sign 'A' for 1 second produces 30 'A' characters.

Design Principles:
- Single Responsibility: Handles temporal stabilization and latch state logic.
- Decoupled from camera hardware and model specifics.
"""

from collections import deque, Counter
from typing import Optional, Tuple


class TemporalDebounceEngine:
    """
    Stabilizes real-time gesture predictions across consecutive video frames.
    
    Attributes:
    -----------
    window_size : int
        Number of consecutive frames required for majority consensus (default: 10).
    confidence_threshold : float
        Minimum prediction confidence required to consider a frame valid (default: 0.85).
    consistency_ratio : float
        Proportion of window frames that must agree on the same gesture (default: 0.80).
    """

    def __init__(
        self,
        window_size: int = 10,
        confidence_threshold: float = 0.85,
        consistency_ratio: float = 0.80,
        absence_timeout_frames: int = 45  # ~1.5 seconds at 30 FPS
    ):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.consistency_ratio = consistency_ratio
        self.absence_timeout_frames = absence_timeout_frames

        # Ring buffer storing recent valid (label, confidence) pairs
        self._history_queue: deque = deque(maxlen=window_size)
        
        # Current locked gesture to prevent repeated emission
        self._latched_sign: Optional[str] = None
        
        # Frame counter tracking how long no hands have been detected
        self._absent_frame_count: int = 0

    def process_frame(
        self,
        predicted_label: Optional[str],
        confidence: float,
        hand_detected: bool
    ) -> Tuple[Optional[str], bool]:
        """
        Processes a single frame's prediction through the temporal filter.

        Parameters:
        -----------
        predicted_label : str or None
            Raw class label predicted by the ML model.
        confidence : float
            Confidence probability in range [0.0, 1.0].
        hand_detected : bool
            Flag indicating whether a hand was localized in the frame.

        Returns:
        --------
        emitted_symbol : Optional[str]
            The confirmed symbol if a new stable gesture was committed, else None.
        is_sentence_boundary : bool
            True if hand has been absent long enough to conclude a word/sentence.
        """
        # Case 1: No hand detected in the current frame
        if not hand_detected or predicted_label is None:
            self._absent_frame_count += 1
            self._history_queue.clear()
            self._latched_sign = None  # Reset latch when hand leaves viewport

            # Trigger sentence boundary when absence exceeds threshold
            is_boundary = (self._absent_frame_count == self.absence_timeout_frames)
            return None, is_boundary

        # Hand is present: reset the absence counter
        self._absent_frame_count = 0

        # Case 2: Hand detected but confidence is below safety cutoff
        if confidence < self.confidence_threshold:
            # Noise / transition frame: do not pollute the history queue
            return None, False

        # Case 3: Valid, high-confidence detection: append to window queue
        self._history_queue.append(predicted_label)

        # Wait until we have accumulated enough frames for statistical consensus
        if len(self._history_queue) < self.window_size:
            return None, False

        # Statistical mode analysis across the temporal window
        counts = Counter(self._history_queue)
        dominant_label, dominant_count = counts.most_common(1)[0]
        consensus_ratio = dominant_count / self.window_size

        # Check if the dominant gesture meets the required consistency threshold
        if consensus_ratio >= self.consistency_ratio:
            # Latch check: If user is still holding the same sign, ignore repetition
            if dominant_label != self._latched_sign:
                # New distinct gesture confirmed!
                self._latched_sign = dominant_label
                return dominant_label, False

        return None, False

    def reset(self) -> None:
        """Resets all internal queues and latch states."""
        self._history_queue.clear()
        self._latched_sign = None
        self._absent_frame_count = 0
