import cv2
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_yellow_colors(image_path, num_colors=6):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([18, 50, 65])  # Lower range for yellow
    upper_yellow = np.array([40, 255, 255])  # Upper range for yellow

    mask = cv2.inRange(image_hsv, lower_yellow, upper_yellow)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    yellow_pixels = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    pixels = yellow_pixels.reshape(-1, 3)
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

    if len(pixels) > 0:
        kmeans = KMeans(n_clusters=min(num_colors, len(pixels)))
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        return colors
    else:
        return np.array([])

def process_images_in_directory(directory_path, output_csv, num_colors=6):
    results = []
    
    # Iterate over all files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(directory_path, file_name)
            colors = extract_yellow_colors(image_path, num_colors)

            row = {'Image Name': file_name}
            if colors.size > 0:
                for i, color in enumerate(colors):
                    row[f'Cluster {i+1} (R)' ] = color[0]
                    row[f'Cluster {i+1} (G)'] = color[1]
                    row[f'Cluster {i+1} (B)'] = color[2]
            else:
                row['Cluster 1 (R)'] = 0
                row['Cluster 1 (G)'] = 0
                row['Cluster 1 (B)'] = 0

            results.append(row)

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# Example usage
directory_path = './validate/validate J'  # Replace with the path to your directory
output_csv = 'validate_yellow_colors_results.csv'
process_images_in_directory(directory_path, output_csv, num_colors=6)
