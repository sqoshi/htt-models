import logging
import os
import pickle
import random
from pathlib import Path

import cv2
from mediapipe.python.solutions.hands import Hands
from sklearn.model_selection import train_test_split


class ASLHandsBoxesSelectedProcessor:
    def __init__(self):
        self.hands = Hands(
            max_num_hands=1,
            static_image_mode=True,
            min_detection_confidence=0.3,
        )

    def load(self, source):
        all_images = [str(file) for file in Path(source).rglob("*") if file.is_file()]
        random.shuffle(all_images)
        return all_images

    def preprocess(self, img_paths):
        result = {"data": [], "labels": []}
        valid_count = 0

        for i, img_path in enumerate(img_paths):
            data_aux, valid = self.process_image(img_path)
            if valid:
                label = img_path.split(os.path.sep)[-2]
                result["data"].append(data_aux)
                result["labels"].append(label)
                valid_count += 1
            if i % 50 == 0:
                logging.debug(f"Processed image {i + 1} out of {len(img_paths)}.")
        logging.info(f"{valid_count} out of {len(img_paths)} images were valid.")
        self.save(result, "data/asl_hands_cut.pickle")
        return result

    def save(self, data, destination):
        with open(destination, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"Data saved to {destination}.")

    def process_image(self, img_path):
        """Detect, crop, and resize the hand in the image."""
        img = cv2.imread(img_path)

        if img is None:
            logging.warning(f"Failed to load image: {img_path}")
            return None, False

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            h, w, _ = img.shape
            x_min = int(
                min([lm.x for lm in results.multi_hand_landmarks[0].landmark]) * w
            )
            y_min = int(
                min([lm.y for lm in results.multi_hand_landmarks[0].landmark]) * h
            )
            x_max = int(
                max([lm.x for lm in results.multi_hand_landmarks[0].landmark]) * w
            )
            y_max = int(
                max([lm.y for lm in results.multi_hand_landmarks[0].landmark]) * h
            )

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            if x_max <= x_min or y_max <= y_min:
                logging.warning(f"Invalid bounding box detected in image: {img_path}")
                return None, False

            hand_region = img[y_min:y_max, x_min:x_max]

            if hand_region.size == 0:
                logging.warning(f"Empty hand region detected in image: {img_path}")
                return None, False

            hand_resized = cv2.resize(hand_region, (28, 28))
            hand_gray = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2GRAY)
            hand_normalized = hand_gray / 255.0
            return hand_normalized.flatten(), True

        return None, False

    def split(self, data, labels, test_size=0.2):
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=test_size, shuffle=True, stratify=labels
        )
        logging.info("Data split into training and test sets.")
        return x_train, x_test, y_train, y_test
