import os

import cv2
import mediapipe as mp


def preprocess_and_save_images(
    root_dir, output_dir, img_size=(224, 224), margin_factor=0.2
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    )

    for person in os.listdir(root_dir):
        person_dir = os.path.join(root_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for label in os.listdir(person_dir):
            label_dir = os.path.join(person_dir, label)
            if not os.path.isdir(label_dir):
                continue

            output_label_dir = os.path.join(output_dir, person, label)
            os.makedirs(output_label_dir, exist_ok=True)

            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    continue

                # Preprocess the image
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Detect hands using MediaPipe
                results = hands.process(image_rgb)
                if not results.multi_hand_landmarks:
                    # Skip this image if no hand is detected
                    print(f"No hand detected in {img_path}, skipping...")
                    continue

                # Hand detected, proceed with cropping and saving
                hand_landmarks = results.multi_hand_landmarks[0]
                image_height, image_width, _ = image.shape
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]

                xmin = int(min(x_coords) * image_width)
                ymin = int(min(y_coords) * image_height)
                xmax = int(max(x_coords) * image_width)
                ymax = int(max(y_coords) * image_height)

                box_width = xmax - xmin
                box_height = ymax - ymin
                margin_x = int(margin_factor * box_width)
                margin_y = int(margin_factor * box_height)

                xmin = max(xmin - margin_x, 0)
                ymin = max(ymin - margin_y, 0)
                xmax = min(xmax + margin_x, image_width)
                ymax = min(ymax + margin_y, image_height)

                cropped_image = image_rgb[ymin:ymax, xmin:xmax]
                if cropped_image.size == 0:
                    processed_image = cv2.resize(image_rgb, img_size)
                else:
                    processed_image = cv2.resize(cropped_image, img_size)

                # Save processed image
                output_img_path = os.path.join(output_label_dir, img_name)
                cv2.imwrite(
                    output_img_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                )

    hands.close()


if __name__ == "__main__":
    root_dir = "/home/piotr/Documents/htt/images"
    output_dir = "/home/piotr/Documents/htt/images224224"
    preprocess_and_save_images(root_dir, output_dir, margin_factor=0.2)
