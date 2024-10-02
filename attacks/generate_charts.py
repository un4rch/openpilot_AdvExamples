import pickle
from PIL import Image
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import imageio

adversarial_dir = "/home/ikerlan/Unai/openpilot/selfdrive/modeld/adversarial"
algorithm = "patches_gradients"
patches_dir = adversarial_dir+"/"+algorithm+"/npy"
images_dir = adversarial_dir+"/"+algorithm+"/png"
all_patches = []
all_data = []


def create_gif(image_list, gif_path, duration=0.1):
    """
    Create a GIF from a list of images.

    Parameters:
    - image_list: List of numpy arrays representing images.
    - gif_path: Output path for the GIF file.
    - duration: Duration for each frame in the GIF.
    """
    # Ensure the images are in uint8 format
    image_list = [img.astype(np.uint8) for img in image_list]
    
    # Save images as GIF
    imageio.mimsave(gif_path, image_list, duration=duration)

# save the camera frames as PNG
def save_image(image_data, file_name):
  """Function to save an image file."""
  try:
    if image_data.dtype != np.uint8:
      image_data = (image_data * 255).astype(np.uint8) # fixed: Error, could not save image:  Cannot handle this data type: (1, 1, 3), <f4
      # Solution: Specifically, it appears that the array's dtype is float32 (indicated by <f4>), which is not directly supported by the PIL library for image saving. Converting the array to a supported dtype, such as uint8, should resolve the issue
    img_save = Image.fromarray(image_data)
    img_save.save(file_name)
    print(f"Image successfully saved: {file_name}")
  except Exception as e:
    print("Error, could not save image: ", e)

# Convert numpy patches in pkl files to png
filenames_with_numbers = []
for filename in os.listdir(patches_dir):
  if filename.endswith('.npy'):
    match = re.search(r'\d+', filename)
    if match:
        number = int(match.group())
        filenames_with_numbers.append((number, filename))

filenames_with_numbers.sort()

for number, filename in filenames_with_numbers:
    with open(os.path.join(patches_dir, filename), 'rb') as file:
        #loaded_data = pickle.load(file)
        loaded_data = np.load(file)
        all_patches.append(loaded_data)
        patch = loaded_data
        save_image(patch, os.path.join(images_dir, f"patch_{number}.png"))

# Generate chart of patches performance
if os.path.exists(adversarial_dir+"/"+algorithm+"/distances.pkl"):
  with open(adversarial_dir+"/"+algorithm+"/distances.pkl", "rb") as file:
    all_data = pickle.load(file)
dists = [patch[0]-patch[1] for patch in all_data]
confs = [patch[2] for patch in all_data]
#losses = [patch[3] for patch in all_data]

plt.plot(dists, linestyle='-', color='b')  # Blue line
plt.plot(dists, marker='o', linestyle='None', color='r', markersize=3) # Red markers
#plt.plot(losses, linestyle='-', color='r')  # Blue line
#plt.plot(losses, marker='o', linestyle='None', color='r', markersize=3) # Red markers
plt.title('Detected distance with Adversarial Example')
plt.xlabel('Epochs')
plt.ylabel('Detected distance')
plt.savefig('distances.png')

plt.clf()
#plt.axhline(y=0, color='grey', linestyle='--', linewidth=1) # Horizontal line
plt.plot(confs, linestyle='-', color='g')  # Blue line
plt.plot(confs, marker='o', linestyle='None', color='r', markersize=3) # Red markers
plt.title('Detection confidence by Openpilot')
plt.xlabel('Epochs')
plt.ylabel('Confidence')
plt.savefig('confs.png')

create_gif(all_patches, adversarial_dir+"/patch_evolution.gif", 0.01)