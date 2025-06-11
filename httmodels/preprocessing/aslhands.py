"""ASL hands dataset processor."""

import logging
import os
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
from mediapipe.python.solutions.hands import Hands

from httmodels.preprocessing.base import ImageProcessor


class ASLHandsProcessor(ImageProcessor):
    """Processor for ASL hands dataset."""

    def __init__(
        self,
        image_size=(28, 28),
        max_hands=1,
        min_detection_confidence=0.3,
        apply_augmentation=False,
    ):
        """Initialize ASL hands processor.

        Args:
            image_size: Size to resize images to (height, width)
            max_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            apply_augmentation: Whether to apply data augmentation
        """
        self.image_size = image_size
        self.apply_augmentation = apply_augmentation
        self.hands = Hands(
            max_num_hands=max_hands,
            static_image_mode=True,
            min_detection_confidence=min_detection_confidence,
        )

    def load(self, source):
        """Load ASL hands dataset from directory.

        Args:
            source: Directory containing ASL hand images

        Returns:
            List of image paths
        """
        logging.info(f"Loading ASL hands dataset from {source}")
        all_images = [str(file) for file in Path(source).rglob("*") if file.is_file()]
        random.shuffle(all_images)
        logging.info(f"Found {len(all_images)} images")
        return all_images

    def process_image(self, img_path):
        """Process a single image for hand detection and cropping.

        Args:
            img_path: Path to image file

        Returns:
            Tuple of (processed_image, is_valid)
        """
        img = cv2.imread(img_path)

        if img is None:
            return None, False

        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return None, False

        # Get hand bounding box
        h, w, _ = img.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0

        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Crop and resize
        cropped = img[y_min:y_max, x_min:x_max]

        if cropped.size == 0:
            return None, False

        resized = cv2.resize(cropped, self.image_size)
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        return gray, True

    def preprocess(self, img_paths):
        """Preprocess ASL hands dataset.

        Args:
            img_paths: List of image paths

        Returns:
            Dictionary with processed data and labels
        """
        result = {"data": [], "labels": []}
        valid_count = 0

        for i, img_path in enumerate(img_paths):
            processed_img, valid = self.process_image(img_path)

            if valid:
                # Extract label from directory name
                label = img_path.split(os.path.sep)[-2]

                # Original image
                result["data"].append(processed_img)
                result["labels"].append(label)
                valid_count += 1

                # Apply augmentation if enabled
                if self.apply_augmentation:
                    # Horizontal flip
                    flipped = cv2.flip(processed_img, 1)
                    result["data"].append(flipped)
                    result["labels"].append(label)

                    # Rotation
                    rows, cols = processed_img.shape
                    for angle in [5, -5, 10, -10]:
                        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                        rotated = cv2.warpAffine(processed_img, M, (cols, rows))
                        result["data"].append(rotated)
                        result["labels"].append(label)

                    # Add small random noise
                    noisy = processed_img.copy()
                    noisy = noisy.astype(np.float32)
                    noisy += np.random.normal(0, 5, noisy.shape).astype(np.float32)
                    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
                    result["data"].append(noisy)
                    result["labels"].append(label)

            if i % 50 == 0:
                logging.info(f"Processed {i + 1}/{len(img_paths)} images")

        logging.info(f"Successfully processed {valid_count}/{len(img_paths)} images")
        if self.apply_augmentation:
            logging.info(f"After augmentation: {len(result['data'])} total samples")
        return result

    def save(self, data, destination):
        """Save preprocessed data.

        Args:
            data: Preprocessed data dictionary
            destination: Destination path

        Returns:
            None
        """
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        logging.info(f"Saving preprocessed ASL hands data to {destination}")

        with open(destination, "wb") as f:
            pickle.dump(data, f)

    def split(self, data, labels, test_size=0.2):
        """Split data into train and test sets.

        Args:
            data: Input data
            labels: Input labels
            test_size: Proportion of data to use for testing

        Returns:
            x_train, x_test, y_train, y_test
        """
        return super().split(data, labels, test_size)
