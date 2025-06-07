import os
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from tqdm import tqdm

class FaceEmbedder:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.inception = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
    def get_embedding(self, img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            face_tensor = self.mtcnn(img)
            if face_tensor is not None:
                face_tensor = face_tensor.to(self.device)
                with torch.no_grad():
                    return self.inception(face_tensor.unsqueeze(0))[0].cpu().numpy()
            print(f"Лицо не обнаружено: {img_path}")
        except Exception as e:
            print(f"Ошибка при обработке {img_path}: {str(e)}")
        return None

def update_user_embedding(username):
    """Обновляет эмбеддинг для конкретного пользователя"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "database")
    faces_path = os.path.join(db_path, "faces", username)
    embeddings_path = os.path.join(db_path, "embeddings")
    
    if not os.path.exists(faces_path):
        print(f"Папка пользователя {username} не найдена")
        return False

    embedder = FaceEmbedder()
    embeddings = []
    
    for file_name in tqdm(os.listdir(faces_path), desc=f"Обновление {username}"):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(faces_path, file_name)
            embedding = embedder.get_embedding(img_path)
            if embedding is not None:
                embeddings.append(embedding)
    
    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        os.makedirs(embeddings_path, exist_ok=True)
        np.save(os.path.join(embeddings_path, f"{username}.npy"), avg_embedding)
        return True
    
    print(f"Не удалось обновить эмбеддинг для {username}")
    return False

def get_embeddings():
    """Получает все эмбеддинги из базы"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "database")
    faces_path = os.path.join(db_path, "faces")
    embeddings_path = os.path.join(db_path, "embeddings")
    
    os.makedirs(embeddings_path, exist_ok=True)
    embedder = FaceEmbedder()
    embeddings = {}

    for person_name in tqdm(os.listdir(faces_path), desc="Обработка пользователей"):
        person_path = os.path.join(faces_path, person_name)
        if os.path.isdir(person_path):
            embedding_file = os.path.join(embeddings_path, f"{person_name}.npy")
            embeddings[person_name] = np.load(embedding_file) if os.path.exists(embedding_file) else None
    
    return embeddings