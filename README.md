# SIGNificant

SIGNificant bridges the communication gap between people with vocal disabilities and the world around them by making sign language easier to understand.

## Features

- Computer vision for translating signs to words using American Sign Language (ASL)
- Machine learning for sign language recognition
- OpenAI Realtime API for instant generation of suggestions of sentences
- Text-to-speech upon selection of sentence

## Installation

### Prerequisites
- Python 3.12 (required for compatibility with Mediapipe)

### Setup Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/CrestyDY/CodeJam.git
   cd CodeJam
   ```

2. Create a virtual environment and install uv for package management:
   ```bash
   python -m venv .venv
   source .venv/bin/activate # Linux
   .venv/Scripts/activate # Windows
   # Sorry mac users
   pip install uv
   ```



3. Install dependencies:
   ```bash
   uv sync
   ```

4. Add your OpenAI API key inside of `etc/.env` following the template given by `etc/template.env`

## Usage

### Training Your Own Dataset

You can run and train your own dataset for sign language:

1. Modify the json files from `src/config` to modify a specific id's word/letter/number
2. Modify the camera indices in both `src/util/collects_imgs.py` (line 85) and `src/util/inference_classifier.py` (line 33)
3. Run the following commands in sequence:

   **Collect training images:**
   ```bash
   python src/util/collect_imgs.py
   # Use --help to see optional arguments
   python src/util/collect_imgs.py --help
   ```

   **Create datasets:**
   ```bash
   python src/util/create_dataset_models.py
   # Use --help to see optional arguments
   python src/util/create_dataset_models.py --help
   ```

   **Train the models:**
   ```bash
   python src/util/train_classifier.py
   # Use --help to see optional arguments
   python src/util/train_classifier.py --help
   ```

   **Test inference:**
   ```bash
   python src/util/inference_classifier.py
   ```
   You will now be able to test whether your hand gestures get detected for the right text conversion.

### Running the Flask Server

You can also run the Flask server to have the app running locally on your browser:

1. Modify your camera inside of `src/app.py` (line 545) if the default settings do not work
2. Start the server:
   ```bash
   python src/app.py
   ```
3. Open `localhost:8000` in your browser to see the app
