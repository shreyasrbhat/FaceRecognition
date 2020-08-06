from facenet_pytorch import MTCNN

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20, keep_all=True,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device)
    return mtcnn