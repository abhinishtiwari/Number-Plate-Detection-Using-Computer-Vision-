import cv2
import argparse
import os
import sys
from pathlib import Path

# Add package directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.detector import PlateDetector
from src.ocr_engine import OCREngine
from src.utils import draw_detection_box, log_detection_to_csv

def process_image(image_path, output_path=None, method="hybrid", ocr_engine_type="auto"):
    """Processes a single input image and detects license plates."""
    if not os.path.exists(image_path):
        print(f"[-] Error: Image path '{image_path}' does not exist.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"[-] Error: Unable to read image '{image_path}'.")
        return

    detector = PlateDetector()
    ocr = OCREngine(engine_type=ocr_engine_type)

    print(f"[+] Processing image: {image_path}...")
    detections = detector.detect_plates(image, method=method)

    if not detections:
        print("[-] No license plates detected.")
        return

    print(f"[+] Found {len(detections)} candidate license plate(s).")
    annotated_image = image.copy()

    for idx, (x, y, w, h, roi, enhanced_roi) in enumerate(detections, 1):
        plate_text, conf = ocr.extract_text(enhanced_roi)
        display_label = plate_text if plate_text else f"Plate #{idx}"
        
        print(f"  -> Plate #{idx}: Text='{plate_text}' (Conf: {conf:.2f}) at Box=({x}, {y}, {w}, {h})")
        
        # Annotate image
        annotated_image = draw_detection_box(annotated_image, (x, y, w, h), label=display_label, confidence=conf)
        
        # Log to CSV
        if plate_text:
            log_detection_to_csv(plate_text, confidence=conf)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated_image)
        print(f"[+] Saved annotated result to: {output_path}")

    return annotated_image

def process_video(video_source, output_path=None, method="hybrid", ocr_engine_type="auto"):
    """Processes a video file or live webcam feed."""
    cap = cv2.VideoCapture(int(video_source) if str(video_source).isdigit() else video_source)
    if not cap.isOpened():
        print(f"[-] Error: Could not open video source '{video_source}'.")
        return

    detector = PlateDetector()
    ocr = OCREngine(engine_type=ocr_engine_type)

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print("[+] Press 'q' to stop video stream.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Process every 2nd frame for performance optimization
        if frame_count % 2 == 0:
            detections = detector.detect_plates(frame, method=method)
            for (x, y, w, h, roi, enhanced_roi) in detections:
                plate_text, conf = ocr.extract_text(enhanced_roi)
                label = plate_text if plate_text else "Plate"
                frame = draw_detection_box(frame, (x, y, w, h), label=label, confidence=conf)

        if writer:
            writer.write(frame)

        cv2.imshow("Number Plate Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[+] Video processing finished.")

def main():
    parser = argparse.ArgumentParser(description="Automatic Number Plate Detection (ANPR) System")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--video", type=str, help="Path to input video file or webcam index (0)")
    parser.add_argument("--output", type=str, help="Path to save annotated output file")
    parser.add_argument("--method", type=str, choices=["cascade", "contour", "hybrid"], default="hybrid", help="Detection method")
    parser.add_argument("--ocr", type=str, choices=["auto", "tesseract", "easyocr"], default="auto", help="OCR Engine")

    args = parser.parse_args()

    if args.image:
        process_image(args.image, output_path=args.output, method=args.method, ocr_engine_type=args.ocr)
    elif args.video:
        process_video(args.video, output_path=args.output, method=args.method, ocr_engine_type=args.ocr)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
