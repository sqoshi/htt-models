import cv2
import numpy as np
from mediapipe.python.solutions.hands import Hands


class HandLandmarkTransformer:

    def __init__(self):
        self.hands = Hands(
            max_num_hands=1,
            static_image_mode=True,
            min_detection_confidence=0.3,
        )

    def transform(self, img):
        data_aux = []
        x_, y_ = [], []
        img_rgb = cv2.cvtColor(img.numpy(), cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            return np.array(data_aux, dtype=np.float32)
        return None
