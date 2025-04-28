import layoutparser as lp
import pytesseract
import cv2
import time
import argparse
import matplotlib.pyplot as plt

def extract_text_from_blocks(image, layout):
    extracted_text = []
    
    crop_times = []
    ocr_times = []
    postprocess_times = []

    for block in layout:
        x1, y1, x2, y2 = map(int, block.coordinates)

        # Time cropping
        start_crop = time.perf_counter()
        cropped_image = image[y1:y2, x1:x2]
        crop_times.append(time.perf_counter() - start_crop)

        # Time OCR
        start_ocr = time.perf_counter()
        text = pytesseract.image_to_string(cropped_image)
        ocr_times.append(time.perf_counter() - start_ocr)

        # Time postprocessing
        start_post = time.perf_counter()
        clean_text = text.strip()
        extracted_text.append(clean_text)
        postprocess_times.append(time.perf_counter() - start_post)

    timings = {
        'crop_time_total': sum(crop_times),
        'ocr_time_total': sum(ocr_times),
        'postprocess_time_total': sum(postprocess_times),
        'total_blocks': len(layout),
    }

    return extracted_text, timings

def benchmark_layoutparser(image_path, model_name="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config", trials=5):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Initialize model
    model = lp.Detectron2LayoutModel(
        config_path=model_name,
	model_path='/Users/rileycox/PubLayNet-faster_rcnn_R_50_FPN_3x/model_final.pth',
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
    )

    total_crop_time = 0.0
    total_ocr_time = 0.0
    total_postprocess_time = 0.0
    total_detection_time = 0.0
    n_trials = 0

    for trial in range(trials):
        print(f"Running trial {trial + 1}...")

        # Detect layout
        start_detect = time.perf_counter()
        layout = model.detect(image)
        detect_time = time.perf_counter() - start_detect
        total_detection_time += detect_time

        # Extract text with detailed timing
        extracted_text, timing_info = extract_text_from_blocks(image, layout)

        total_crop_time += timing_info['crop_time_total']
        total_ocr_time += timing_info['ocr_time_total']
        total_postprocess_time += timing_info['postprocess_time_total']

        n_trials += 1

    # Compute averages
    avg_crop_time = total_crop_time / n_trials
    avg_ocr_time = total_ocr_time / n_trials
    avg_postprocess_time = total_postprocess_time / n_trials
    avg_detection_time = total_detection_time / n_trials

    # Summarize
    print("\n=== Benchmark Results ===")
    print(f"Average Layout Detection Time: {avg_detection_time:.4f} seconds")
    print(f"Average Cropping Time: {avg_crop_time:.4f} seconds")
    print(f"Average OCR Time: {avg_ocr_time:.4f} seconds")
    print(f"Average Postprocessing Time: {avg_postprocess_time:.4f} seconds")

    plot_bottleneck_chart(avg_crop_time, avg_ocr_time, avg_postprocess_time)

def plot_bottleneck_chart(avg_crop_time, avg_ocr_time, avg_postprocess_time):
    labels = ['Cropping', 'OCR', 'Postprocessing']
    sizes = [avg_crop_time, avg_ocr_time, avg_postprocess_time]

    total = sum(sizes)
    percentages = [100 * s / total for s in sizes]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, percentages, color=['skyblue', 'salmon', 'lightgreen'])

    # Label bars with percentage
    for bar, percentage in zip(bars, percentages):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{percentage:.1f}%', ha='center', va='bottom')

    ax.set_ylabel('Percentage of OCR Pipeline Time (%)')
    ax.set_title('OCR Bottleneck Breakdown')
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_name", type=str, default="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config", help="LayoutParser model name")
    parser.add_argument("--trials", type=int, default=5, help="Number of benchmark trials")
    args = parser.parse_args()

    benchmark_layoutparser(args.image_path, model_name=args.model_name, trials=args.trials)

