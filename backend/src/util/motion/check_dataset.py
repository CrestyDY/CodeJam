"""
Check motion dataset statistics and provide feedback.
"""
import pickle
import numpy as np
import sys
import os

def check_dataset(pickle_path='motion_data.pickle'):
    """Check and report on motion dataset."""
    
    if not os.path.exists(pickle_path):
        print(f"❌ Dataset file not found: {pickle_path}")
        print("\nPlease collect training data first:")
        print("  python create_dataset_motion.py --mode interactive --hands 1")
        return False
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        sequences = data['sequences']
        labels = data['labels']
        
        print("="*60)
        print("MOTION DATASET STATISTICS")
        print("="*60)
        
        print(f"\nTotal sequences: {len(sequences)}")
        
        if len(sequences) == 0:
            print("\n❌ Dataset is empty!")
            print("\nPlease collect training data:")
            print("  python create_dataset_motion.py --mode interactive --hands 1")
            return False
        
        # Get unique classes and counts
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        print(f"Number of classes: {len(unique_labels)}")
        print(f"Sequence length: {data.get('sequence_length', 'unknown')}")
        
        if sequences:
            seq_array = np.array(sequences[0])
            feature_dim = seq_array.shape[-1]
            hands = 1 if feature_dim == 42 else 2
            print(f"Hands detected: {hands}")
            print(f"Feature dimension: {feature_dim}")
        
        print("\n" + "="*60)
        print("CLASS DISTRIBUTION")
        print("="*60)
        
        min_count = min(counts)
        max_count = max(counts)
        
        for label, count in zip(unique_labels, counts):
            bar_length = int((count / max_count) * 30)
            bar = "█" * bar_length
            
            # Determine status
            if count < 2:
                status = "❌ TOO FEW"
            elif count < 10:
                status = "⚠️  LOW"
            elif count < 20:
                status = "✓ OK"
            else:
                status = "✓✓ GOOD"
            
            print(f"  {label:15s} {count:3d} {bar} {status}")
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        ready_to_train = True
        
        # Check overall size
        if len(sequences) < 4:
            print("\n❌ NOT ENOUGH DATA TO TRAIN")
            print(f"   Current: {len(sequences)} sequences")
            print(f"   Minimum: 4 sequences (absolute minimum)")
            print(f"   Recommended: 50+ sequences")
            ready_to_train = False
        elif min_count < 2:
            print("\n❌ SOME CLASSES HAVE TOO FEW EXAMPLES")
            print(f"   Minimum samples per class: {min_count}")
            print(f"   Required: At least 2 per class for train/test split")
            ready_to_train = False
        elif len(sequences) < 10:
            print("\n⚠️  DATASET IS VERY SMALL")
            print(f"   Current: {len(sequences)} sequences")
            print(f"   Recommended: 50+ sequences (10+ per class)")
            print("\n   You can train, but accuracy may be low.")
            print("   Consider collecting more data for better results.")
        elif min_count < 10:
            print("\n⚠️  SOME CLASSES HAVE FEW EXAMPLES")
            print(f"   Minimum samples per class: {min_count}")
            print(f"   Recommended: 10+ per class")
            print("\n   You can train, but consider collecting more data")
            print("   for classes with few examples.")
        else:
            print("\n✓ Dataset looks good!")
            print(f"   Total sequences: {len(sequences)}")
            print(f"   Samples per class: {min_count} to {max_count}")
            if len(sequences) < 50:
                print("\n   💡 Tip: 50+ sequences will give better accuracy")
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        
        if ready_to_train:
            print("\n✓ Ready to train!")
            print("\n  Train the model:")
            print("    python train_motion_classifier.py --data motion_data.pickle --model-type lstm")
        else:
            print("\n📝 Collect more data:")
            print("    python create_dataset_motion.py --mode interactive --hands 1")
            print("\n   Focus on classes with few examples:")
            for label, count in zip(unique_labels, counts):
                if count < 2:
                    print(f"    - {label}: needs {2 - count} more")
        
        print("\n" + "="*60 + "\n")
        
        return ready_to_train
        
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check motion dataset statistics")
    parser.add_argument(
        "--data",
        type=str,
        default="motion_data.pickle",
        help="Path to motion dataset pickle file"
    )
    
    args = parser.parse_args()
    
    success = check_dataset(args.data)
    sys.exit(0 if success else 1)

