import os
from PIL import Image

# Folder containing images
folder_path = "./new/Normal Class/validate"
image_size = (200, 200)

# Process each image in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Process only images
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)  # Open image
        img = img.resize(image_size, Image.LANCZOS)  # Resize image
        img.save(img_path)  # Overwrite the original image
        print(f"Resized: {filename}")

print("✅ All images have been resized and saved successfully!")
