"""
Train LSTM-based classifier for motion-based sign language recognition.
"""
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models")  # Shared models directory

# Ensure models directory exists
os.makedirs(MODEL_PATH, exist_ok=True)


def load_motion_data(pickle_path):
    """Load motion dataset from pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data['sequences'], data['labels']


def build_lstm_model(sequence_length, feature_dim, num_classes, lstm_units=64):
    """
    Build LSTM model for motion recognition.
    
    Args:
        sequence_length: Number of frames per sequence
        feature_dim: Number of features per frame (42 for 1 hand, 84 for 2 hands)
        num_classes: Number of sign classes
        lstm_units: Number of LSTM units
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, feature_dim)),
        
        # First LSTM layer with return sequences
        layers.LSTM(lstm_units, return_sequences=True, dropout=0.2),
        layers.BatchNormalization(),
        
        # Second LSTM layer
        layers.LSTM(lstm_units, dropout=0.2),
        layers.BatchNormalization(),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_gru_model(sequence_length, feature_dim, num_classes, gru_units=64):
    """
    Build GRU model for motion recognition (alternative to LSTM).
    GRU is often faster and can work just as well.
    """
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, feature_dim)),
        
        layers.GRU(gru_units, return_sequences=True, dropout=0.2),
        layers.BatchNormalization(),
        
        layers.GRU(gru_units, dropout=0.2),
        layers.BatchNormalization(),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_conv_lstm_model(sequence_length, feature_dim, num_classes):
    """
    Build Conv1D + LSTM hybrid model.
    Conv1D can capture local temporal patterns before LSTM processes them.
    """
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, feature_dim)),
        
        # 1D Convolution to extract local temporal features
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        # LSTM to process temporal sequences
        layers.LSTM(64, return_sequences=True, dropout=0.2),
        layers.LSTM(64, dropout=0.2),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_motion_classifier(data_path, model_type='lstm', model_name='motion_model'):
    """
    Train a motion-based sign classifier.
    
    Args:
        data_path: Path to motion dataset pickle file
        model_type: Type of model ('lstm', 'gru', or 'conv_lstm')
        model_name: Name for saving the model
    """
    print(f"\n=== Training Motion Classifier ===")
    print(f"Data: {data_path}")
    print(f"Model type: {model_type}")
    
    # Load data
    sequences, labels = load_motion_data(data_path)
    
    if not sequences:
        print("❌ No data found!")
        print("\nPlease collect training data first:")
        print("  python create_dataset_motion.py --mode interactive --hands 1")
        return None, None, None

    if len(sequences) < 4:
        print(f"❌ Not enough data! Found {len(sequences)} sequences, need at least 4.")
        print("\nPlease collect more training data:")
        print("  - Minimum: 4 sequences (2 train, 2 test)")
        print("  - Recommended: 50+ sequences (10+ per class)")
        return None, None, None

    print(f"Loaded {len(sequences)} sequences")
    
    # Convert to numpy arrays
    X = np.array(sequences)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    print(f"Sequence shape: {X.shape}")
    print(f"Classes: {label_encoder.classes_}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for cls, count in zip(label_encoder.classes_[unique], counts):
        print(f"  {cls}: {count} samples")

    min_samples = np.min(counts)

    # Split data - disable stratify if dataset is too small
    if min_samples < 2 or len(sequences) < 10:
        print("\n⚠️  Warning: Dataset is small. Using simple split without stratification.")
        print(f"   Minimum samples per class: {min_samples}")
        print(f"   Total samples: {len(sequences)}")
        print(f"   Recommendation: Collect at least 10 examples per class (50+ total)")

        if len(sequences) < 4:
            print("\n❌ Error: Not enough data to train!")
            print("   Need at least 4 sequences total (2 for training, 2 for testing)")
            return None, None, None

        # Use simple split without stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
    else:
        # Use stratified split for larger datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Build model
    sequence_length = X.shape[1]
    feature_dim = X.shape[2]
    num_classes = len(label_encoder.classes_)
    
    if model_type == 'lstm':
        model = build_lstm_model(sequence_length, feature_dim, num_classes)
    elif model_type == 'gru':
        model = build_gru_model(sequence_length, feature_dim, num_classes)
    elif model_type == 'conv_lstm':
        model = build_conv_lstm_model(sequence_length, feature_dim, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print("\nModel Summary:")
    model.summary()
    
    # Adjust training parameters based on dataset size
    if len(X_train) < 10:
        epochs = 50
        batch_size = 2
        patience = 5
        print("\n⚠️  Small dataset detected - using adjusted training parameters:")
        print(f"   Epochs: {epochs}, Batch size: {batch_size}, Patience: {patience}")
    elif len(X_train) < 50:
        epochs = 75
        batch_size = 8
        patience = 8
    else:
        epochs = 100
        batch_size = 32
        patience = 10

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODEL_PATH, f"{model_name}_best.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("\n=== Training ===")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n=== Evaluation ===")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save model and metadata
    model_save_path = os.path.join(MODEL_PATH, f"{model_name}.h5")
    model.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    
    # Save label encoder and metadata
    metadata = {
        'label_encoder': label_encoder,
        'classes': label_encoder.classes_.tolist(),
        'sequence_length': sequence_length,
        'feature_dim': feature_dim,
        'num_classes': num_classes,
        'model_type': model_type,
        'test_accuracy': test_acc,
        'test_loss': test_loss
    }
    
    metadata_path = os.path.join(MODEL_PATH, f"{model_name}_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to: {metadata_path}")
    
    return model, history, metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train motion-based sign classifier")
    parser.add_argument(
        "--data",
        type=str,
        default="motion_data.pickle",
        help="Path to motion dataset pickle file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=['lstm', 'gru', 'conv_lstm'],
        default='lstm',
        help="Type of model to train"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="motion_model",
        help="Name for the saved model"
    )
    
    args = parser.parse_args()
    
    # Train model
    train_motion_classifier(
        data_path=args.data,
        model_type=args.model_type,
        model_name=args.name
    )
    
    print("\n✓ Training complete!")

