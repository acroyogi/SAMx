import torch
import numpy as np
from PIL import Image, ImageDraw

# Load your PIL image
image = Image.open("jesstest.png")

# Example tensors (replace these with your actual tensor data)
# Assuming the tensor represents bounding boxes or regions (in normalized coordinates)
# Tensor should be in the format [x1, y1, x2, y2] for bounding boxes
# tensor_1 = torch.tensor([0.1, 0.1, 0.4, 0.4])
# tensor_2 = torch.tensor([0.5, 0.5, 0.8, 0.8])
# tensor_3 = torch.tensor([0.2, 0.6, 0.5, 0.9])

# CATS
# my_object = {'scores': torch.tensor([0.4786, 0.4379, 0.4760]), 
#   'labels': ['a cat', 'a cat', 'a remote control'], 
#   'boxes': torch.tensor([
#         [344.6985,  23.1084, 637.1810, 374.2747],
#         [ 12.2695,  51.9101, 316.8557, 472.4342],
#         [ 38.5873,  70.0089, 176.7759, 118.1748]
#     ])
# }

# JESS
my_object = {'scores': torch.tensor([0.8442, 0.4941, 0.4673, 0.5106, 0.4693, 0.4538, 0.4247, 0.4181]),
    'labels': ['a ball', 'a person', 'a person', 'a person', 'a person', 'a person', 'a person', 'a person'], 
    'boxes': torch.tensor([[ 100.1498,  344.5766,  167.2417,  417.2546],
        [ 300.9248,  995.4282,  458.6963, 1537.4375],
        [ 598.4009, 1237.8290, 1411.5615, 2429.3679],
        [  30.5240, 1007.3854,  243.4566, 1451.5353],
        [1054.9377, 1055.6158, 1179.5193, 1457.4084],
        [ 183.8696,  443.6328, 1429.4788, 1534.7891],
        [ 897.7634,  975.5735, 1037.7122, 1507.3246],
        [ 454.1052,  998.6483,  575.7560, 1421.5927]
    ])
}

# List of tensors
tensors = my_object["boxes"]
labels = my_object["labels"]

# Convert the image size to get the dimensions
image_width, image_height = image.size

# Create a drawing context
draw = ImageDraw.Draw(image)

# Superimpose each tensor as a rectangle on the image
for tensor, label in zip(tensors, labels):
    # Denormalize the tensor coordinates to image dimensions
    x1, y1, x2, y2 = tensor # * torch.tensor([image_width, image_height, image_width, image_height])
    # Draw the rectangle on the image
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    # Position for the label text (top-left corner of the bounding box)
    label_position = (x1, y1)
    # Draw the label text
    draw.text(label_position, label, fill="white", font_size=24)


# Save the image with tensors superimposed
image.save("output_image_with_tensors_v003.png")

# Display the image (optional)
image.show()
