"""Video frame sampling plus MTCNN face detection."""
from pathlib import Path
import cv2
from facenet_pytorch import MTCNN

def extract_faces(video_path: str, output_dir: str, every_n_frames=15):
    """Sample a video; save its largest detected face from every Nth frame."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    detector = MTCNN(keep_all=True, device="cuda" if __import__('torch').cuda.is_available() else "cpu")
    cap, frame_no, saved = cv2.VideoCapture(video_path), 0, []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        if frame_no % every_n_frames == 0:
            boxes, _ = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if boxes is not None:
                x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                h, w = frame.shape[:2]; x1,y1,x2,y2 = map(int, (max(0,x1),max(0,y1),min(w,x2),min(h,y2)))
                face = frame[y1:y2, x1:x2]
                if face.size:
                    path = output / f"face_{frame_no:06d}.jpg"; cv2.imwrite(str(path), face); saved.append(str(path))
        frame_no += 1
    cap.release(); return saved
