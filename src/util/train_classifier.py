import pickle
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'data')
MODEL_PATH = os.path.join(SCRIPT_DIR, "models")

def train_model(dataset_name, model_name, description):
    """Train a single model from a dataset pickle file"""
    try:
        data_dict = pickle.load(open(os.path.join(DATA_PATH, dataset_name), 'rb'))
        data = np.asarray(data_dict['data'])
        labels = np.asarray(data_dict['labels'])

        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, shuffle=True, stratify=labels
        )

        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        score = accuracy_score(y_predict, y_test)

        print(f'✓ {description}: {score * 100:.2f}% accuracy')

        # Save model
        with open(os.path.join(MODEL_PATH, model_name), 'wb') as f:
            pickle.dump({'model': model}, f)

        return True
    except FileNotFoundError:
        print(f'✗ {description}: Dataset file not found ({dataset_name})')
        return False
    except Exception as e:
        print(f'✗ {description}: Error training model - {e}')
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL classifier models")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["letters", "numbers", "words_one_hand", "words_two_hands", "all"],
        default="all",
        help="Which model to train: 'letters', 'numbers', 'words_one_hand', 'words_two_hands', or 'all' (default)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Training ASL Classifier Models")
    print("=" * 60)

    # Train models based on mode
    if args.mode in ["letters", "all"]:
        train_model("letters_data.pickle", "model_letters.p", "LETTERS (one-hand)")

    if args.mode in ["numbers", "all"]:
        train_model("numbers_data.pickle", "model_numbers.p", "NUMBERS (one-hand)")

    if args.mode in ["words_one_hand", "all"]:
        train_model("words_one_hand_data.pickle", "model_words_one_hand.p", "WORDS (one-hand)")

    if args.mode in ["words_two_hands", "all"]:
        train_model("words_two_hands_data.pickle", "model_words_two_hands.p", "WORDS (two-hands)")

    print("=" * 60)
    print("Training complete!")
    print("=" * 60)

