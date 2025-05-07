from paddleocr import PaddleOCR, draw_ocr
import time

# Initialize
ocr = PaddleOCR(use_angle_cls=True, lang='en')
img_path = '/Users/rileycox/Desktop/image.png'

# Step 1: Total Time
total_start = time.perf_counter()
result = ocr.ocr(img_path, cls=True)
total_end = time.perf_counter()

print(f"🔎 Total OCR Time: {total_end - total_start:.4f} sec")

# Step 2: Estimate Detection vs Recognition Timing
# How? Count boxes (detection count ~ fast), recognition scales with # of boxes

num_boxes = sum([len(line) for line in result])
print(f"📝 Detected {num_boxes} text boxes")

# Step 3: Simulate finer profiling by batching multiple images
imgs = [img_path] * 5  # Simulate batch of 5 images

det_times = []
rec_times = []

for img in imgs:
    t0 = time.perf_counter()
    result = ocr.ocr(img, cls=True)
    t1 = time.perf_counter()
    det_times.append(t1 - t0)  # Total for now; we approximate below

# Approximate:
#  - Detection is fairly fixed per image
#  - Recognition scales with box count
avg_time = sum(det_times) / len(det_times)
print(f"📊 Avg Time per Image: {avg_time:.4f} sec")


