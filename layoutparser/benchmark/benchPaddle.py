import time
from paddleocr import PaddleOCR
from PIL import Image

# Initialize PaddleOCR with GPU support (set gpu=True if you have a GPU)
ocr = PaddleOCR(use_angle_cls=True, lang='en', gpu=True)  # Set gpu=True if GPU is available

# Function to benchmark each part of OCR pipeline
def benchmark_ocr_pipeline(image_path):
    # Load the image
    img = Image.open(image_path)
    
    # Stage 1: Preprocessing (resize, etc.)
    start_time = time.time()
    # PaddleOCR internally handles this step during OCR processing, so we'll just note its timing
    # No explicit preprocessing call in this case
    preprocessing_time = time.time() - start_time
    
    # Stage 2: Text Detection
    start_time = time.time()
    detection_result = ocr.ocr(image_path, cls=False)  # Disable classification to just focus on detection
    detection_time = time.time() - start_time
    
    # Stage 3: Text Recognition
    start_time = time.time()
    recognition_result = ocr.ocr(image_path, cls=True)  # Now use cls=True to run full OCR
    recognition_time = time.time() - start_time
    
    # Stage 4: Postprocessing (if applicable, but here it's not explicit)
    postprocessing_time = 0  # There's no explicit postprocessing, so this is just a placeholder

    # Print the time taken for each stage
    print(f"Preprocessing Time: {preprocessing_time:.4f} seconds")
    print(f"Text Detection Time: {detection_time:.4f} seconds")
    print(f"Text Recognition Time: {recognition_time:.4f} seconds")
    print(f"Postprocessing Time: {postprocessing_time:.4f} seconds")
    print(f"Total OCR Time: {preprocessing_time + detection_time + recognition_time + postprocessing_time:.4f} seconds")

# Specify the image to benchmark
image_path = '/Users/rileycox/Desktop/image.png'

# Run the benchmark
benchmark_ocr_pipeline(image_path)

