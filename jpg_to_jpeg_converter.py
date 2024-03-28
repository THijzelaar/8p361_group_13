import os
from PIL import Image

# Function to convert all .jpg files in a folder and its subfolders to .jpeg files
def convert_jpg_to_jpeg(source_folder, destination_folder):
    # Walk through all the folders and files in the source folder
    for root, dirs, files in os.walk(source_folder):
        # Determine the relative path from the source folder to the current folder
        relative_path = os.path.relpath(root, source_folder)
        # Create the corresponding folder structure in the destination folder
        dest_path = os.path.join(destination_folder, relative_path)
        os.makedirs(dest_path, exist_ok=True)

        # Iterate over all files in the current folder
        for file in files:
            if file.lower().endswith('.jpg'):
                # Construct source and destination file paths
                source_file = os.path.join(root, file)
                dest_file = os.path.join(dest_path, os.path.splitext(file)[0] + '.jpeg')

                # Open and convert the image
                with Image.open(source_file) as img:
                    img.convert('RGB').save(dest_file, 'JPEG')

# Example usage:
source_folder = './Data'
destination_folder = './Data_jpeg'

convert_jpg_to_jpeg(source_folder, destination_folder)