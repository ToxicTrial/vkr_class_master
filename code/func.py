import os
import cv2
import numpy as np
import torch
from torchvision import transforms

def load_models():
    from facenet_pytorch import MTCNN, InceptionResnetV1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = MTCNN(keep_all=True, device=device)
    inception = InceptionResnetV1(pretrained='vggface2').to(device).eval()
    return device, detector, inception

def camera_connect(index):
    try:
        cap = cv2.VideoCapture(index)
        print("Камера найдена")
    except Exception:
        print("Камера не найдена")
    return cap
 
def detect_faces(img, embeddings, detector, device, inception):
    boxes, probs = detector.detect(img)
    faces = []
    if boxes is not None:
        for box, prob in zip(boxes, probs):
            x1, y1, x2, y2 = map(int, box)
            face_img = img[y1:y2, x1:x2]
            name, score = recognize_face(face_img, embeddings, device, inception)
            faces.append({
                'box': box.tolist(),
                'confidence': float(prob),
                'name': name,
                'score': float(score)
            })
    return faces

def load_embeddings(path):
    embeddings = {}
    for filename in os.listdir(path):
        if filename.endswith(".npy"):
            name = os.path.splitext(filename)[0]
            data = np.load(os.path.join(path, filename))
            embeddings[name] = data
    print(F"Найдены эмбеддинги: {embeddings}")
    return embeddings

def get_embedding(face_img, device, inception):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160, 160)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    if face_img is None:
        return None

    face_tensor = transform(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = inception(face_tensor)[0].cpu().numpy()
    return embedding

def recognize_face(face, embeddings, device, inception):
    test_embedding = get_embedding(face, device, inception)
    if test_embedding is None or test_embedding.size == 0 or not embeddings:
        return "Лицо не обнаружено", -1
    
    best_match = "Неизвестный"
    best_score = -1

    for person_name, person_embeddings in embeddings.items():
        similarities = cosine_similarity(test_embedding, person_embeddings)
        max_similarity = np.max(similarities)

        if max_similarity > best_score:
            best_score = max_similarity
            best_match = person_name

    print(best_match, best_score)
    return best_match, best_score

def cosine_similarity(vec, ref_vecs):
    vec = vec / np.linalg.norm(vec)
    if ref_vecs.ndim == 1:
        ref_vecs = ref_vecs / np.linalg.norm(ref_vecs)
        return np.dot(vec, ref_vecs)
    else:
        ref_vecs = ref_vecs / np.linalg.norm(ref_vecs, axis=1, keepdims=True)
        return np.dot(ref_vecs, vec)