import os
import pickle
import cv2
import mediapipe as mp
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, 'data')
PICKLE_PATH = os.path.join(SCRIPT_DIR, 'data.pickle')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3,
    max_num_hands=2
)


# ONE-HAND DATASET BUILDER

def build_dataset_one_hand(data_dir):
    """
    Expects images in:
      data_dir/
        0/
          img_0.jpg, img_1.jpg, ...
        1/
        2/
        ...
    Uses images where EXACTLY ONE hand is detected.
    Produces 42 features per sample (21 landmarks × (x,y)).
    """
    data = []
    labels = []

    if not os.path.exists(data_dir):
        print(f"[ONE-HAND] Directory does not exist: {data_dir}")
        return data, labels

    for dir_ in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, dir_)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)

            img = cv2.imread(img_path)
            if img is None:
                print(f"[ONE-HAND] Warning: cannot read {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                continue

            # require exactly one hand
            if len(results.multi_hand_landmarks) != 1:
                continue

            hand_landmarks = results.multi_hand_landmarks[0]

            data_aux = []
            x_ = []
            y_ = []

            # collect coords
            for lm in hand_landmarks.landmark:
                x = lm.x
                y = lm.y
                x_.append(x)
                y_.append(y)

            min_x = min(x_)
            min_y = min(y_)

            # same normalization as inference: (x - min_x, y - min_y)
            for lm in hand_landmarks.landmark:
                x = lm.x
                y = lm.y
                data_aux.append(x - min_x)
                data_aux.append(y - min_y)

            if len(data_aux) != 42:
                print(f"[ONE-HAND] Skip {img_path}: {len(data_aux)} features (need 42)")
                continue

            data.append(data_aux)
            labels.append(dir_)

    print(f"[ONE-HAND] Built {len(data)} samples from {data_dir}")
    return data, labels


# TWO-HANDS DATASET BUILDER

def build_dataset_two_hands(data_dir):
    """
    Expects images in:
      data_dir/
        0/
        1/
        ...
    Uses images where EXACTLY TWO hands are detected.
    Produces 84 features per sample:
      2 hands × 21 landmarks × (x,y)
    Each hand is normalized by (min_x_hand, min_y_hand),
    exactly like in your two-hand inference branch.
    """
    data = []
    labels = []

    if not os.path.exists(data_dir):
        print(f"[TWO-HANDS] Directory does not exist: {data_dir}")
        return data, labels

    for dir_ in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, dir_)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)

            img = cv2.imread(img_path)
            if img is None:
                print(f"[TWO-HANDS] Warning: cannot read {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                continue

            # require exactly two hands
            if len(results.multi_hand_landmarks) != 2:
                continue

            data_aux = []

            # process each hand separately, per-hand normalization
            for hand_landmarks in results.multi_hand_landmarks:
                x_hand = []
                y_hand = []

                for lm in hand_landmarks.landmark:
                    x = lm.x
                    y = lm.y
                    x_hand.append(x)
                    y_hand.append(y)

                min_x_hand = min(x_hand)
                min_y_hand = min(y_hand)

                for lm in hand_landmarks.landmark:
                    x = lm.x
                    y = lm.y
                    data_aux.append(x - min_x_hand)
                    data_aux.append(y - min_y_hand)

            if len(data_aux) != 84:
                print(f"[TWO-HANDS] Skip {img_path}: {len(data_aux)} features (need 84)")
                continue

            data.append(data_aux)
            labels.append(dir_)

    print(f"[TWO-HANDS] Built {len(data)} samples from {data_dir}")
    return data, labels


# SAVE HELPER function
def save_dataset(data, labels, out_path):
    with open(out_path, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)
    print(f"Saved {len(data)} samples to {out_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ASL datasets for one-hand and/or two-hands models")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["one", "two", "both"],
        default="both",
        help="Which dataset to build: 'one' (one-hand), 'two' (two-hands), or 'both' (default)"
    )

    args = parser.parse_args()

    ONE_HAND_DIR = os.path.join(BASE_DIR, "one_hand")
    TWO_HANDS_DIR = os.path.join(BASE_DIR, "two_hands")

    # Build datasets based on mode
    if args.mode in ["one", "both"]:
        print(f"\n=== Building ONE-HAND dataset from {ONE_HAND_DIR} ===")
        data_one, labels_one = build_dataset_one_hand(ONE_HAND_DIR)
        if data_one:
            save_dataset(data_one, labels_one, "one_hand_data.pickle")
        else:
            print("[ONE-HAND] No data collected. Check your directory and images.")

    if args.mode in ["two", "both"]:
        print(f"\n=== Building TWO-HANDS dataset from {TWO_HANDS_DIR} ===")
        data_two, labels_two = build_dataset_two_hands(TWO_HANDS_DIR)
        if data_two:
            save_dataset(data_two, labels_two, "two_hands_data.pickle")
        else:
            print("[TWO-HANDS] No data collected. Check your directory and images.")

    print("\n✓ Done building datasets.")
