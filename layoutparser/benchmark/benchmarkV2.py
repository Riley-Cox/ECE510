import layoutparser as lp
import time
import argparse
import cv2
import pytesseract
import numpy as np


def extract_text_from_blocks(image, layout):
    """Extract text from detected layout blocks using pytesseract."""
    extracted_text = []
    for block in layout:
        # Get block coordinates (bounding box)
        x1, y1, x2, y2 = block.coordinates
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cropped_image = image[y1:y2, x1:x2]

        # OCR to extract text
        text = pytesseract.image_to_string(cropped_image)
        extracted_text.append(text)
    return extracted_text


def run_single_trial(model, image):
    timings = {}

    # Inference timing
    start_time = time.perf_counter()
    layout = model.detect(image)
    timings['inference'] = time.perf_counter() - start_time

    # Post-processing timing
    start_time = time.perf_counter()
    _ = [block.type for block in layout]  # Just to make sure layout processing is done
    timings['post_processing'] = time.perf_counter() - start_time

    # Extract text
    start_time = time.perf_counter()
    extracted_text = extract_text_from_blocks(image, layout)
    timings['text_extraction'] = time.perf_counter() - start_time

    return timings, extracted_text


def benchmark_layoutparser(image_path, model_name="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config", device="cpu", trials=5):
    timings_list = []
    extracted_text_list = []

    # Load model
    model_load_start = time.perf_counter()
    model = lp.Detectron2LayoutModel(
        config_path=model_name,
	model_path='/Users/rileycox/PubLayNet-faster_rcnn_R_50_FPN_3x/model_final.pth',
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
    )
    model.model.model.to(device)  # Move model to correct device
    model_load_time = time.perf_counter() - model_load_start
    print(f"Model loaded in {model_load_time:.4f} seconds.")

    # Load image once
    image_load_start = time.perf_counter()
    image = cv2.imread(image_path)
    image_load_time = time.perf_counter() - image_load_start
    print(f"Image loaded in {image_load_time:.4f} seconds.")

    # Warm-up (important for GPU)
    _ = model.detect(image)

    print(f"\nRunning {trials} trials...\n")
    for i in range(trials):
        trial_timings, extracted_text = run_single_trial(model, image)
        timings_list.append(trial_timings)
        extracted_text_list.append(extracted_text)
        print(f"Trial {i+1}: Inference {trial_timings['inference']:.4f}s, Post-processing {trial_timings['post_processing']:.4f}s, Text extraction {trial_timings['text_extraction']:.4f}s")

    # Aggregate results
    inference_times = [t['inference'] for t in timings_list]
    post_processing_times = [t['post_processing'] for t in timings_list]
    text_extraction_times = [t['text_extraction'] for t in timings_list]

    print("\n=== Benchmark Summary ===")
    print(f"Model Loading Time   : {model_load_time:.4f} seconds")
    print(f"Image Loading Time   : {image_load_time:.4f} seconds")
    print(f"Inference Time       : {np.mean(inference_times):.4f} ± {np.std(inference_times):.4f} seconds")
    print(f"Post-processing Time : {np.mean(post_processing_times):.4f} ± {np.std(post_processing_times):.4f} seconds")
    print(f"Text Extraction Time : {np.mean(text_extraction_times):.4f} ± {np.std(text_extraction_times):.4f} seconds")
    print(f"Average Total per Run: {np.mean(np.array(inference_times) + np.array(post_processing_times) + np.array(text_extraction_times)):.4f} seconds")

    return {
        "model_loading": model_load_time,
        "image_loading": image_load_time,
        "inference_mean": np.mean(inference_times),
        "inference_std": np.std(inference_times),
        "post_processing_mean": np.mean(post_processing_times),
        "post_processing_std": np.std(post_processing_times),
        "text_extraction_mean": np.mean(text_extraction_times),
        "text_extraction_std": np.std(text_extraction_times),
    }, extracted_text_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LayoutParser model performance with text extraction.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run model on: 'cpu' or 'cuda'")
    parser.add_argument("--model_name", type=str, default="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                        help="LayoutParser model config URL or path.")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials to average timings.")
    args = parser.parse_args()

    results, extracted_text_list = benchmark_layoutparser(args.image_path, model_name=args.model_name, device=args.device, trials=args.trials)


