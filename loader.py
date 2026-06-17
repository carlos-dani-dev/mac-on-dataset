import sys
import io
import os
from pathlib import Path
from functools import wraps
from dotenv import load_dotenv
load_dotenv()

import cv2 as cv
from tqdm import tqdm
import PIL as pil
from PIL import Image

import pickle
import pandas as pd
import numpy as np

import tensorflow as tf
from facenet_pytorch import MTCNN
from keras_facenet import FaceNet

from concurrent.futures import ThreadPoolExecutor

MTCNN_MODEL = MTCNN(keep_all=True, device='cuda')

def get_total_images(dataset_path, valid_extensions=[".jpg", ".jpeg", ".png"]):
    
    if valid_extensions is not None:
        valid_extensions = [ext.lower() for ext in valid_extensions]
    total=0
    for path in Path(dataset_path).rglob("*"):
        if not path.is_file():
            continue

        if valid_extensions:
            if path.suffix.lower() not in valid_extensions:
                continue
        total+=1
    return total

TOTAL_IMAGES = get_total_images(dataset_path="./dataset")
DATASET_PATH = os.getenv("DATASET_PATH")



def capture_output(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout

    return wrapper


@capture_output
def crop_face(original_image, boxes, probs):
    
    if boxes is not None and len(boxes) > 0:
        alpha=40
        best_idx = np.argmax(probs)
        x1, y1, x2, y2 = [int(coord) for coord in boxes[best_idx]]

        x1-=alpha
        y1-=alpha
        x2+=alpha
        y2+=alpha

        w, h = original_image.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cropped_face = original_image.crop((x1, y1, x2, y2))

        return np.asarray(cropped_face)
    else:
        return None


def redim_face(cropped_face_array, required_size=(160, 160)):
    
    cropped_face_image = pil.Image.fromarray(cropped_face_array)
    face_image = cropped_face_image.convert('RGB')
    resized_face_image = face_image.resize(required_size)
    
    return np.asarray(resized_face_image)


@capture_output
def get_embeddings(facenet, resized_image_array_batch):

    embedding = facenet.embeddings(resized_image_array_batch)
    return embedding

import os
from pathlib import Path

def load_faces_generator(dataset_path,batch_size=10, valid_extensions=[".jpg", ".jpeg", ".png"]):

    if valid_extensions is not None:
        valid_extensions = [ext.lower() for ext in valid_extensions]

    batch_files = []

    for path in Path(dataset_path).rglob("*"):

        if not path.is_file():
            continue

        if valid_extensions:
            if path.suffix.lower() not in valid_extensions:
                continue
        filepath = str(path)

        batch_files.append(filepath)

        if len(batch_files) == batch_size:
            yield batch_files
            batch_files = []

    if batch_files:
        yield batch_files


def load_single_image(filename):
    img = cv.imread(filename)
    
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    #img_resized = cv.resize(img_rgb, (720, 720))
    
    return Image.fromarray(img_rgb)


def load_face_embeddings():
    facenet = FaceNet()

    chaves = ["filename", "face_embedding"]

    df_dict = {chave: [] for chave in chaves}
    print(TOTAL_IMAGES)
    pbar = tqdm(total=TOTAL_IMAGES, desc="Processamento de imagens em lotes", ncols=100)
    with ThreadPoolExecutor(max_workers=18) as executor:
        for image_filenames_batch in load_faces_generator(DATASET_PATH):
            resized_image_array_batch = []
            valid_index = []

            #original_images_batch = list(executor.map(load_single_image, image_filenames_batch))
            original_images_batch = [load_single_image(img) for img in image_filenames_batch]
            
            boxes_batch, probs_batch = MTCNN_MODEL.detect(original_images_batch)

            for i in range(len(original_images_batch)):
                img_boxes = boxes_batch[i] if boxes_batch is not None else None
                img_probs = probs_batch[i] if probs_batch is not None else None

                cropped_image_array = crop_face(original_images_batch[i], img_boxes, img_probs)
                if cropped_image_array is None: continue
                
                resized_image_array = redim_face(cropped_image_array)
                resized_image_array_batch.append(resized_image_array)
                valid_index.append(i)
            
            if not resized_image_array_batch:
                pbar.update(len(image_filenames_batch))
                continue
                
            image_embeddings_batch=get_embeddings(facenet, resized_image_array_batch)

            for idx, i in enumerate(valid_index):
                df_dict["filename"].append(image_filenames_batch[i])
                df_dict["face_embedding"].append(image_embeddings_batch[idx])

            pbar.update(len(image_filenames_batch))

    pbar.close()

    df = pd.DataFrame(df_dict)
    df.to_pickle("./embeddings/embeddings.pkl")

            
if __name__ == "__main__":

    load_face_embeddings()