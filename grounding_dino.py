import requests

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"
# device = "cuda"
device = "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(image_url, stream=True).raw)
img_url = "jesstest.png"
image = Image.open(img_url).convert("RGB")

# Check for cats and remote controls
text = "person. yoga mat. gun. tree. ball."

inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)
print(results)

my_object = results[0]

# List of tensors
tensors = my_object["boxes"]
labels = my_object["labels"]

# Convert the image size to get the dimensions
image_width, image_height = image.size

# Create a drawing context
draw = ImageDraw.Draw(image)

label_object = {
    "person" : "red",
    "yoga mat" : "purple",
    "tree" : "green",
    "ball" : "blue",
}

# Superimpose each tensor as a rectangle on the image
for tensor, label in zip(tensors, labels):
    # Denormalize the tensor coordinates to image dimensions
    x1, y1, x2, y2 = tensor # * torch.tensor([image_width, image_height, image_width, image_height])
    # Draw the rectangle on the image
    draw.rectangle([x1, y1, x2, y2], outline=label_object[label], width=3)
    # Position for the label text (top-left corner of the bounding box)
    label_position = (x1, y1)
    # Draw the label text
    draw.text(label_position, label, fill="white", font_size=36, stroke_width=1)


# Save the image with tensors superimposed
image.save("output_image_with_tensors_v003.png")

# Display the image (optional)
image.show()
