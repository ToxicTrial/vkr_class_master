import os
import cv2
from facenet_pytorch import MTCNN
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def capture_faces(username):
    """Захват изображений лица через камеру"""
    detector = MTCNN(keep_all=True, device=device)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    faces_path = os.path.join(base_dir, "database", "faces", username)
    os.makedirs(faces_path, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False, 0

    i = len([f for f in os.listdir(faces_path) if f.endswith(('.jpg', '.png'))])
    max_images = 100
    
    while i < max_images:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = detector.detect(frame_rgb)
        
        if boxes is not None:
            x1, y1, x2, y2 = map(int, boxes[0])
            face_img = frame[y1:y2, x1:x2]
            
            filename = f"{username}_{i+1}.jpg"
            cv2.imwrite(os.path.join(faces_path, filename), face_img)
            i += 1

    cap.release()
    return True, i