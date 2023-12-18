import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load a pre-trained DeepLabV3+ model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

# Define a function to perform segmentation on an image
def segment_person(image_path):
    input_image = Image.open(image_path)
    
    # Preprocess the input image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # Run inference
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    print(output)
    # Process the output mask to visualize different parts
    mask = output.argmax(0).cpu().numpy()
    # Define random unique colors for different parts
    colors = {
        0: [255, 255, 255],      # Background (Red)
        1: [0, 255, 0],      # Torso or Clothes (Green)
        2: [0, 0, 255],      # Head (Blue)
        3: [255, 255, 0],    # Face (Yellow)
        4: [255, 0, 255],    # Left Hand (Magenta)
        5: [0, 255, 255],    # Right Hand (Cyan)
        # Add more unique colors and corresponding class IDs as needed
    }
    
    segmented_image = np.zeros_like(input_image)
    for class_id, color in colors.items():
        segmented_image[mask == class_id] = color
    
    return segmented_image

# Path to the folder containing the images
image_path = "test_img/she.jpg"
segmented_image = segment_person(image_path)

# Display the segmented image
plt.imshow(segmented_image)
plt.axis('off')
plt.show()
