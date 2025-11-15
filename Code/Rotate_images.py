import cv2, os

def flip_images():
	gest_folder = "gestures"
	if not os.path.exists(gest_folder):
		print(f"Error: '{gest_folder}' directory not found.")
		return
	
	images_labels = []
	images = []
	labels = []
	
	for g_id in os.listdir(gest_folder):
		gesture_path = os.path.join(gest_folder, g_id)
		if not os.path.isdir(gesture_path):
			continue  # Skip if it's not a directory
		
		print(f"Processing gesture folder: {g_id}")
		processed = 0
		skipped = 0
		
		for i in range(1200):
			path = os.path.join(gest_folder, g_id, str(i+1)+".jpg")
			new_path = os.path.join(gest_folder, g_id, str(i+1+1200)+".jpg")
			
			# Check if source file exists
			if not os.path.exists(path):
				skipped += 1
				continue
			
			# Check if destination already exists (skip if it does)
			if os.path.exists(new_path):
				skipped += 1
				continue
			
			# Read the image
			img = cv2.imread(path, 0)
			if img is None:
				print(f"Warning: Could not read {path}")
				skipped += 1
				continue
			
			# Flip the image
			img = cv2.flip(img, 1)
			
			# Write the flipped image
			if cv2.imwrite(new_path, img):
				processed += 1
			else:
				print(f"Warning: Could not write {new_path}")
				skipped += 1
		
		print(f"  Gesture {g_id}: Processed {processed} images, Skipped {skipped} images")
	
	print("Done flipping images!")

flip_images()
