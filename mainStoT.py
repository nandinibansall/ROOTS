from pdf2image import convert_from_path
import pytesseract as pt
import cv2
import numpy as np 

# img=cv2.imread("img.jpg")

pt.pytesseract.tesseract_cmd="C:\\Users\\peehu\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"

pages = convert_from_path("pdf2.pdf", dpi=300,poppler_path="C:\\poppler-25.07.0\\poppler-25.07.0\\Library\\bin")

with open("output2.txt", "w", encoding="utf-8") as f:
    for page_num, page in enumerate(pages):
        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        text = pt.image_to_string(gray)
    
        f.write(f"--- Page {page_num + 1} ---\n")
        f.write(text + "\n\n")

print("Text extraction complete. Saved to output.txt")

