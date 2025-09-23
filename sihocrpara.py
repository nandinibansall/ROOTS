#!/usr/bin/env python
# coding: utf-8

# In[6]:


from PIL import Image
import numpy as np
import cv2
import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

image_path = "test.jpg"   
output_dir = "lines"         

os.makedirs(output_dir, exist_ok=True)


img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

proj = np.sum(thresh, axis=1)

lines = []
in_line = False
start = 0
threshold = np.max(proj) * 0.05  

for i, val in enumerate(proj):
    if val > threshold and not in_line:
        in_line = True
        start = i
    elif val <= threshold and in_line:
        in_line = False
        end = i
        lines.append((start, end))

line_images = []
for idx, (s, e) in enumerate(lines):
    line_img = img[s:e, :]
    line_path = os.path.join(output_dir, f"line_{idx+1}.png")
    cv2.imwrite(line_path, line_img)
    line_images.append(line_path)

print(f"✅ Saved {len(line_images)} line images into '{output_dir}'")


print("\n🔹 Running OCR with TrOCR...\n")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for idx, line_path in enumerate(line_images, 1):
    image = Image.open(line_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"Line {idx}: {generated_text}")





