import layoutparser as lp
from pdf2image import convert_from_path
import numpy as np

# Step 1: Convert PDF to image(s)
images = convert_from_path("/Users/rileycox/Documents/paper-image.pdf", dpi=300)

# Step 2: Load layout detection model
model = lp.models.Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    model_path='/Users/rileycox/PubLayNet-faster_rcnn_R_50_FPN_3x/model_final.pth',
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
)

# Step 3: Load OCR Agent
ocr_agent = lp.TesseractAgent(languages='eng')  # You can add 'deu', 'fra', etc.

# Step 4: Detect layout and extract text
for i, image in enumerate(images):
    image_np = np.array(image)
    layout = model.detect(image_np)

    # OCR only within text-like blocks
    text_blocks = [b for b in layout if b.type in ['Text', 'Title']]

    for block in text_blocks:
        sub_image = block.crop_image(image_np)  # Crop the image to just the block
        text = ocr_agent.detect(sub_image)
        print(f"[Page {i+1}] Block {block.id if hasattr(block, 'id') else ''} → {text}")

