"""
Sentence and Word Assembler Module.

This module manages the text-building buffer for fingerspelling and word signs.
It stitches individual characters into words, handles backspaces, space boundaries,
and whole-word insertions.

Design Principles:
- Single Responsibility: Manages the grammar and string buffer state.
- Separation of Concerns: Independent of both video frame capture and ML models.
"""

from typing import Tuple


class SentenceAssembler:
    """
    Assembles real-time streams of characters and word signs into complete sentences.
    """

    def __init__(self):
        self._sentence_tokens: list = []  # List of completed words
        self._current_word: str = ""       # Current word being fingerspelled

    def ingest_symbol(self, symbol: str) -> Tuple[str, bool]:
        """
        Ingests a debounced symbol from the ML engine.

        Parameters:
        -----------
        symbol : str
            The recognized token (e.g. 'A', 'space', 'del', 'HELLO', 'YES').

        Returns:
        --------
        display_text : str
            The current active text to display on the subtitle overlay.
        word_completed : bool
            True if a word or space boundary was finalized.
        """
        symbol_lower = symbol.lower()

        # Case 1: Space Gesture
        if symbol_lower == "space":
            word_completed = False
            if self._current_word:
                self._sentence_tokens.append(self._current_word)
                self._current_word = ""
                word_completed = True
            return self.get_display_text(), word_completed

        # Case 2: Delete / Backspace Gesture
        elif symbol_lower in ("del", "delete", "backspace"):
            if len(self._current_word) > 0:
                self._current_word = self._current_word[:-1]
            elif len(self._sentence_tokens) > 0:
                # Pop the last completed word back into editing
                self._current_word = self._sentence_tokens.pop()
            return self.get_display_text(), False

        # Case 3: Whole Word Sign (e.g., 'HELLO', 'YES', 'NO', 'HELP')
        elif len(symbol) > 1 and symbol_lower != "nothing":
            # Commit any existing partial word first
            if self._current_word:
                self._sentence_tokens.append(self._current_word)
                self._current_word = ""
            # Append the whole word token directly
            self._sentence_tokens.append(symbol.upper())
            return self.get_display_text(), True

        # Case 4: Single Alphabet / Digit Character ('A' through 'Z', '0' through '9')
        elif len(symbol) == 1 and symbol.isalnum():
            self._current_word += symbol.upper()
            return self.get_display_text(), False

        # Case 5: 'nothing' or unrecognized tokens are ignored
        return self.get_display_text(), False

    def finalize_and_flush(self) -> str:
        """
        Concludes the current sentence, returns the full string for TTS,
        and resets the internal buffer.

        Returns:
        --------
        str: The full sentence string that was flushed.
        """
        full_text = self.get_display_text()
        self.clear()
        return full_text

    def get_display_text(self) -> str:
        """
        Returns the formatted string for real-time subtitle rendering.
        """
        parts = list(self._sentence_tokens)
        if self._current_word:
            parts.append(self._current_word)
        return " ".join(parts)

    def clear(self) -> None:
        """Clears all sentence tokens and active word buffers."""
        self._sentence_tokens.clear()
        self._current_word = ""
