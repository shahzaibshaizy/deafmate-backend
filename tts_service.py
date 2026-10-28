"""
Text-to-Speech (TTS) Service Module.

This module provides non-blocking, offline speech synthesis for real-time
sign language translation.

Architectural Pattern:
- Producer-Consumer Pattern using a background daemon thread and Queue.
- Critical Benefit: Calling TTS synchronously blocks the main execution loop,
  causing video frames to freeze. This background worker ensures 30+ FPS video
  stream continuity while speech plays asynchronously.

Design Principles:
- Single Responsibility: Manages speech audio generation and device voice settings.
- Thread-safe: Queues speech requests safely across threads.
"""

import queue
import threading
from typing import Optional
import pyttsx3


class TTSService:
    """
    Asynchronous, non-blocking offline Text-to-Speech engine.
    """

    def __init__(self, speech_rate: int = 160, volume: float = 1.0):
        """
        Initializes the TTS worker thread and queue.

        Parameters:
        -----------
        speech_rate : int
            Words per minute (default: 160, natural conversational speed).
        volume : float
            Audio output volume in range [0.0, 1.0].
        """
        self.speech_rate = speech_rate
        self.volume = volume

        # Thread-safe queue for speech requests
        self._speech_queue: queue.Queue = queue.Queue()
        
        # Flag to control the background worker lifecycle
        self._is_running = True
        
        # Start background worker daemon thread
        self._worker_thread = threading.Thread(
            target=self._process_speech_queue,
            daemon=True,
            name="TTSWorkerThread"
        )
        self._worker_thread.start()

    def _process_speech_queue(self) -> None:
        """
        Background worker loop: initializes engine in its own thread context
        and consumes text phrases from the queue.
        """
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.speech_rate)
            engine.setProperty("volume", self.volume)
        except Exception as e:
            print(f"[TTSService Warning] Could not initialize local TTS engine: {e}")
            return

        while self._is_running:
            try:
                # Wait for text to speak (timeout allows checking _is_running flag)
                text = self._speech_queue.get(timeout=0.5)
                if text:
                    engine.say(text)
                    engine.runAndWait()
                self._speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTSService Error] Speech playback error: {e}")

    def speak(self, text: str) -> None:
        """
        Submits text to be spoken asynchronously.

        Parameters:
        -----------
        text : str
            The sentence or word to voice aloud.
        """
        clean_text = text.strip()
        if clean_text:
            self._speech_queue.put(clean_text)

    def stop(self) -> None:
        """Stops the worker thread safely."""
        self._is_running = False
