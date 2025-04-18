import layoutparser as lp
from pdf2image import convert_from_path
import numpy as np
import cv2
from PIL import Image

# Load model and OCR agent
model = lp.models.Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    model_path='/Users/rileycox/PubLayNet-faster_rcnn_R_50_FPN_3x/model_final.pth',
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
)

ocr_agent = lp.TesseractAgent(languages="eng")

# Convert PDF to images
images = convert_from_path("/Users/rileycox/Documents/paper-image.pdf", dpi=300)

with open("transcript.txt", "w", encoding="utf-8") as transcript_file:
    for i, pil_image in enumerate(images):
        image_np = np.array(pil_image)
        layout = model.detect(image_np)

        # Filter relevant blocks
        blocks = [b for b in layout if b.type in ["Text", "Title", "List", "Table"]]

        transcript_file.write(f"\n--- Page {i+1} ---\n")

        # Extract and save OCR text from each layout region
        for block in blocks:
            sub_image = block.crop_image(image_np)
            text = ocr_agent.detect(sub_image)
            transcript_file.write(text.strip() + "\n")

        print(f"Processed page {i+1}")

