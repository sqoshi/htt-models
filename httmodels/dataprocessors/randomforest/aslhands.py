import logging
import os
import pickle
import random
from pathlib import Path

import cv2
from mediapipe.python.solutions.hands import Hands
from sklearn.model_selection import train_test_split

from httmodels.dataprocessors.abstract import DataProcessor


class ASLHandsProcessor(DataProcessor):
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
        self.save(result, "data/asl_hands.pickle")
        return result

    def process_image(self, img_path):
        data_aux = []
        x_, y_ = [], []
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))
            return data_aux, True

        return None, False

    def save(self, data, destination):
        with open(destination, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"Data saved to {destination}.")

    def split(self, data, labels, test_size=0.2):
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=test_size, shuffle=True, stratify=labels
        )
        logging.info("Data split into training and test sets.")
        return x_train, x_test, y_train, y_test
