"""
Coordinate Normalization Module for Hand Landmarks.

This module provides geometric normalization for MediaPipe hand landmarks.
It ensures that extracted hand skeletons are invariant to:
1. Spatial Translation: Hand position in the camera frame (centered at wrist).
2. Euclidean Scale: Distance from the camera lens (scale normalized to [-1, 1]).

Design Principles Followed:
- Single Responsibility Principle (SRP): Dedicated strictly to geometric transformations.
- DRY (Don't Repeat Yourself): Shared across dataset extraction, offline evaluation,
  and real-time live inference.
"""

from typing import List, Tuple, Union
import numpy as np


class CoordinateNormalizer:
    """
    Normalizes 3D hand keypoints extracted by MediaPipe.
    
    A standard MediaPipe hand produces 21 landmark joints with (x, y, z) coordinates.
    Without normalization, coordinates vary based on whether the user is close,
    far, or standing to the left/right of the camera viewport.
    """

    NUM_LANDMARKS: int = 21
    NUM_COORDINATES_PER_LANDMARK: int = 3
    FEATURE_VECTOR_LENGTH: int = 63  # 21 * 3

    @staticmethod
    def normalize_landmarks(
        raw_landmarks: List[Union[Tuple[float, float, float], object]]
    ) -> np.ndarray:
        """
        Normalizes a collection of 21 hand landmarks.

        Algorithm:
        ----------
        Step 1: Parse coordinates into a float32 NumPy array of shape (21, 3).
        Step 2: Wrist Centering (Translation Invariance)
                Let Landmark 0 be the wrist joint: (x_0, y_0, z_0).
                x'_i = x_i - x_0
                y'_i = y_i - y_0
                z'_i = z_i - z_0  for all i in [0, 20].
        Step 3: Euclidean Scale Invariance (Distance Invariance)
                Compute the maximum Euclidean distance from the origin (wrist):
                d_max = max(sqrt(x'_i^2 + y'_i^2 + z'_i^2)) for all i in [0, 20].
                If d_max > 0, scale all coordinates:
                p''_i = p'_i / d_max
        Step 4: Flatten the (21, 3) matrix into a 1D feature vector of length 63.

        Parameters:
        -----------
        raw_landmarks : List of 21 landmarks, either:
            - Objects with attributes .x, .y, .z (from MediaPipe Landmark objects), or
            - Tuples/lists of (x, y, z) numeric values.

        Returns:
        --------
        np.ndarray: 1D float32 array of shape (63,) containing normalized features.
        """
        if len(raw_landmarks) != CoordinateNormalizer.NUM_LANDMARKS:
            raise ValueError(
                f"Expected exactly {CoordinateNormalizer.NUM_LANDMARKS} landmarks, "
                f"received {len(raw_landmarks)}."
            )

        # 1. Extract raw (x, y, z) coordinates
        coords = np.zeros(
            (CoordinateNormalizer.NUM_LANDMARKS, CoordinateNormalizer.NUM_COORDINATES_PER_LANDMARK),
            dtype=np.float32
        )

        for i, lm in enumerate(raw_landmarks):
            if hasattr(lm, "x") and hasattr(lm, "y"):
                z_val = getattr(lm, "z", 0.0)
                coords[i] = [lm.x, lm.y, z_val]
            else:
                coords[i] = [lm[0], lm[1], lm[2] if len(lm) > 2 else 0.0]

        # 2. Translate coordinates relative to Landmark 0 (Wrist)
        wrist = coords[0].copy()
        centered_coords = coords - wrist

        # 3. Compute Euclidean distances of each landmark from the wrist
        distances = np.linalg.norm(centered_coords, axis=1)
        max_distance = np.max(distances)

        # 4. Scale normalize to ensure invariant hand size
        # Epsilon (1e-6) prevents division by zero in edge cases
        if max_distance > 1e-6:
            normalized_coords = centered_coords / max_distance
        else:
            normalized_coords = centered_coords

        # 5. Flatten to 1D 63-dimensional feature array
        return normalized_coords.flatten()
