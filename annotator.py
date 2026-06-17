import sys
import io
import os
from pathlib import Path
from functools import wraps
from dotenv import load_dotenv
load_dotenv()

import cv2 as cv
from tqdm import tqdm
from PIL import Image

import pickle
import pandas as pd
import numpy as np

import torch
import tensorflow as tf
from facenet_pytorch import MTCNN
from keras_facenet import FaceNet
from tensorflow.keras.models import load_model


MODEL = load_model(os.getenv("MODEL_PATH"))
for layer in MODEL.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
MTCNN_MODEL = MTCNN(keep_all=True, device='cuda')
FACENET = FaceNet()

LE = {}
with open(os.getenv("LE_PATH"), 'rb') as f:
    LE = pickle.load(f)
ATT = ["fitz_type_scale", "male_or_not_male"]    

DF = pd.read_pickle("./embeddings/embeddings.pkl")

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
def get_embeddings(img):
    return FACENET.embeddings(img)


@capture_output
def get_embeddings_batch(img_batch):
    embeddings_batch = FACENET.embeddings(img_batch)
    return np.reshape(embeddings_batch, (-1, 512))

@capture_output
def crop_face(original_image, boxes, probs):
    """
    Extrai a primeira face detectada usando MTCNN (PyTorch).
    Retorna o rosto recortado e redimensionado como array RGB.
    """

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
        return None  # Nenhuma face detectada


def redim_face(cropped_face_array, required_size=(160, 160)):
    
    cropped_face_image = Image.fromarray(cropped_face_array)
    face_image = cropped_face_image.convert('RGB')
    resized_face_image = face_image.resize(required_size)
    
    return np.asarray(resized_face_image)


def extract_face_batch(img_batch, req_size=(160, 160)):

    img_rgb_batch = [i.convert('RGB') for i in img_batch]

    boxes_batch, probs_batch = MTCNN_MODEL.detect(img_rgb_batch)

    detected_faces_idx = []

    img_array_batch = []

    for i in range(len(img_rgb_batch)):
        img_boxes = boxes_batch[i]
        img_probs = probs_batch[i]

        cropped_image_array = crop_face(img_rgb_batch[i], img_boxes, img_probs)
        if cropped_image_array is None: continue
        
        resized_image_array = redim_face(cropped_image_array)
        img_array_batch.append(resized_image_array)
        detected_faces_idx.append(i)
    
    return img_array_batch, detected_faces_idx


def calculate_reliability(probs_array, alpha=0.05):
    x = np.array(probs_array, dtype=np.float32)
    centrality = (1.0 - alpha) * np.mean(x)

    diff_matrix = np.abs(x[:, None] - x[None, :])
    dispersion = alpha * np.mean(diff_matrix)

    reliability = centrality - dispersion
    return round(float(reliability), 5)


def inference_batch(embeddings_batch, atributos, model):

    dict_batch = []

    pred_batch = model(embeddings_batch, training=False)
    pred_batch_np = [p.numpy() for p in pred_batch]

    for idx in range(len(embeddings_batch)):

        dict_ = {}

        total_atributos = len(atributos)
        for j in range(total_atributos):
            if total_atributos == 1:
                probs = pred_batch_np[idx]
            else:
                probs = pred_batch_np[j][idx]

            if tf.is_tensor(probs):
                probs = probs.numpy()

            predicted_class_index=-1

            if probs.ndim == 0 or probs.shape == ():

                predicted_class_index = (probs > 0.5).astype(int)
                
                prob_value = float(probs)
                idx_max = 0 if prob_value <= 0.5 else 1
            elif probs.ndim == 1 and probs.shape[0] == 1:
                
                predicted_class_index = (probs > 0.5).astype(int)
                
                prob_value = float(probs[0])
                idx_max = 0 if prob_value <= 0.5 else 1
            else:

                predicted_class_index = np.argmax(probs, axis=-1).astype(int)

                idx_max = np.argmax(probs)
                prob_value = float(probs[idx_max])
            
            decoded_label = LE[j].inverse_transform([predicted_class_index.item()])[0]
            
            dict_[f"{atributos[j]}_prediction"] = decoded_label
            dict_[f"{atributos[j]}_inference_idx"] = idx_max
            dict_[f"{atributos[j]}_reliability"] = -1
        dict_batch.append(dict_)

    return dict_batch


def batch_model_training_function(model, img_embedding, m):

    img_embedding = np.reshape(img_embedding, (1, 512))
    img_embedding_batch = np.repeat(img_embedding, m, axis=0)

    model_training_prediction_batch = model(img_embedding_batch, training=True)

    dict_training = {}
    for i in range(len(ATT)): 
        probs = model_training_prediction_batch[i]
        if tf.is_tensor(probs):
            probs = probs.numpy()
        dict_training[ATT[i]] = np.squeeze(probs)

    return dict_training


def load_faces_generator(df, batch_size=128):
    
    img_embeddings = df['face_embedding'].values
    img_embeddings = np.vstack(img_embeddings)

    for i in range(0, len(img_embeddings), batch_size):
            yield img_embeddings[i:i+batch_size]


def load_single_image(filename):
    return Image.open(filename).convert("RGB")


def load_embeddings_generator(df, batch_size=128):
    
    img_embeddings = df['face_embedding']
    img_embeddings = np.vstack(img_embeddings)

    for i in range(0, len(img_embeddings), batch_size):
            yield img_embeddings[i:i+batch_size]


def prediction_function(df, model, atributos, m=100):

    for img_embedding_batch in load_embeddings_generator(df, batch_size=128):

        list_dict = []

        dict_inference_batch = inference_batch(img_embedding_batch, atributos, model)

        img_embedding_giant_batch = np.repeat(img_embedding_batch, m, axis=0)
        training_giant_batch = model(img_embedding_giant_batch, training=True)

        #numpy_training_giant_batch = np.array([out.numpy() if tf.is_tensor(out) else out for out in training_giant_batch], dtype=np.float32)

        for i in range(len(img_embedding_batch)):

            current_face_dict = dict_inference_batch[i]
            current_face_embedding = img_embedding_batch[i]

            #batch_dict_training = batch_model_training_function(model, current_face_embedding, m)

            for j, atributo in enumerate(atributos):
                #raw_batch_predictions = batch_dict_training[atributo]
                idx_target = current_face_dict[f"{atributo}_inference_idx"]

                training_pred_start_idx = i*m
                training_pred_end_idx = training_pred_start_idx+m

                # print("inicio: ", training_pred_start_idx)
                # print("fim: ",training_pred_end_idx)
                # print(training_giant_batch.shape)
                probs = training_giant_batch[j][training_pred_start_idx:training_pred_end_idx]
                # print("PROBS: ", probs)
                if tf.is_tensor(probs):
                    probs = probs.numpy()

                raw_batch_predictions = probs
                if raw_batch_predictions.ndim == 1 or (raw_batch_predictions.ndim == 2 and raw_batch_predictions.shape[1] == 1):
                    raw_batch_predictions = np.squeeze(raw_batch_predictions)
                    if idx_target == 0:
                        lista_100_predicoes = 1.0 - raw_batch_predictions
                    else:
                        lista_100_predicoes = raw_batch_predictions
                else:
                    lista_100_predicoes = raw_batch_predictions[:, idx_target]
                
                confiabilidade = calculate_reliability(lista_100_predicoes, alpha=0.05)
                
                current_face_dict[f"{atributo}_reliability"] = confiabilidade

            list_dict.append(current_face_dict)
        
        if list_dict:
            yield list_dict


def prep_dataset(dataset_path):
    extensoes = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

    list_filepaths = pd.DataFrame([
        str(arquivo)
        for arquivo in Path(dataset_path).rglob("*")
        if arquivo.is_file() and arquivo.suffix.lower() in extensoes
    ])


if __name__ == "__main__":

    list_pred = []
    total_imgs = len(DF)
    
    pbar = tqdm(total=len(DF), desc="Processamento de imagens em lotes", ncols=100)
    for pred_batch in prediction_function(DF, MODEL, ATT, m=100):
        list_pred.extend(pred_batch)
        pbar.update(len(pred_batch))
    pbar.close()

    print(f"DATASET COMPLETAMENTE ANOTADO")

    df_ = pd.DataFrame(list_pred)

    for atributo in ATT:
        DF[f"{atributo}_prediction"] = df_[f"{atributo}_prediction"].values
        DF[f"{atributo}_reliability"] = df_[f"{atributo}_reliability"].values
        
    DF.to_pickle(f"./dataset_annotation/dataset_annotated.pkl")
