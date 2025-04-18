import layoutparser as lp
import cv2
from PIL import Image

# Load image (supports .png)
image_path = "/Users/rileycox/Desktop/image.png"
image = cv2.imread(image_path)
if image is None:
	raise FileNotFoundError("Could not load image: check your path!")


# Choose a pre-trained model (PubLayNet, etc.)
model = lp.models.Detectron2LayoutModel(
    config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
    model_path='/Users/rileycox/PubLayNet-faster_rcnn_R_50_FPN_3x/model_final.pth',
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
)
# Detect layout
layout = model.detect(image)

# Draw results
image_with_layout = lp.draw_box(image, layout, box_width=3)

# Show or save
image_with_layout.show()  # Or .save("output.png")

