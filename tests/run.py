import uuid

import cv2
import mediapipe as mp
import torch


class CNNHandService:
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
        )

    def preprocess_image(self, hand_img):
        hand_img_resized = cv2.resize(hand_img, (28, 28))
        hand_img_gray = cv2.cvtColor(hand_img_resized, cv2.COLOR_BGR2GRAY)
        hand_img_normalized = hand_img_gray / 255.0
        hand_img_tensor = (
            torch.tensor(hand_img_normalized, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return hand_img_tensor.to(self.device)

    def detect_hand(self, frame, margin_percentage=0.3):
        frame_rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )  # Convert to RGB for MediaPipe
        results = self.mp_hands.process(frame_rgb)  # Detect the hand

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape

                x_min = int(
                    min([landmark.x for landmark in hand_landmarks.landmark]) * w
                )
                y_min = int(
                    min([landmark.y for landmark in hand_landmarks.landmark]) * h
                )
                x_max = int(
                    max([landmark.x for landmark in hand_landmarks.landmark]) * w
                )
                y_max = int(
                    max([landmark.y for landmark in hand_landmarks.landmark]) * h
                )

                margin_x = int((x_max - x_min) * margin_percentage)
                margin_y = int((y_max - y_min) * margin_percentage)

                x_min = max(0, x_min - margin_x)
                y_min = max(0, y_min - margin_y)
                x_max = min(w, x_max + margin_x)
                y_max = min(h, y_max + margin_y)

                hand_region = frame[y_min:y_max, x_min:x_max]

                cv2.imwrite(f"./trash/{uuid.uuid4()}.jpg", hand_region)

                return hand_region

        return None

    def predict(self, frame):
        hand_region = self.detect_hand(frame)
        if hand_region is not None:
            preprocessed_hand = self.preprocess_image(hand_region)
            with torch.no_grad():
                output = self.model(preprocessed_hand)
                _, predicted_class = torch.max(output, 1)
            return self.letter(predicted_class.item())
        return None

    def letter(self, number: int):
        if not number:
            return ""
        if 1 <= number <= 9:
            return chr(number + ord("A") - 1)
        return chr(number + ord("A"))


def run_realtime_inference(model_service):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predicted_class = model_service.predict(frame)

        if predicted_class is not None:
            print(f"Predicted Class: {predicted_class}")
            cv2.putText(
                frame,
                f"Class: {predicted_class}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "No Hand Detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = torch.load("../models/cnn.pth", map_location=device)

#     model_service = CNNHandService(model=model, device=device)

#     run_realtime_inference(model_service)
