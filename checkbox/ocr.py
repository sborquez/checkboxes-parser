import json
from os import path

import numpy as np
import pandas as pd
from PIL import Image
from pytesseract import Output, image_to_data

from .shapes import *


def save_image(image, filepath):
    """
    save_image guarda la imagen en la ruta dada.
    """
    if path.exists(filepath):
        print("[Info] Sobreescribiendo imagen", filepath)

    if isinstance(image, Image.Image):
        image.save(filepath)
    else:     
        if image.dtype == np.dtype("float64") and image.max() <= 1.0:
            image =  image * 255
        
        cv2.imwrite(filepath, image)


def prepare_image(image_path, target_size=1500, resize=True):
    img = Image.open(image_path)
    W, H = img.size

    if not resize:
        return img, 1
    
    if W > H:
        r = target_size / W
    else:
        r = target_size / H

    img_resized = img.resize((int(W*r), int(H*r)))
    return img_resized, r


def apply_ocr_tesseract(image):
    data = image_to_data(image, output_type=Output.DICT)
    data = pd.DataFrame(data)
    data = data[(pd.to_numeric(data["conf"]) > 80)  & ~(data["text"].isin(["", " ", "O", "|", "/", "  "]))]
    relevants = data["text"].value_counts()[data["text"].value_counts() == 1].index
    data = data[data["text"].isin(relevants)]

    x, y = data[["left", "top"]].values.T
    t = data["text"].values
    
    return x.tolist(), y.tolist(), t.tolist()


def save_to_json(x, y, text, filename):
    data = []
    for xx, yy, tt in zip(x, y, text):
        data.append({
            "x": xx,
            "y": yy,
            "text": tt
        })
    with open(filename, "w") as f:
        json.dump(data, f)


def get_bboxs_from_tesseract(json_path):
    with open(json_path) as f:
        data=json.load(f)  
    bboxs = []
    text = []
    for d  in data:
        points = [d["x"], d["y"]]*4
        bboxs.append(MOCR_BBox(points, 4))
        text.append(d["text"])

    return bboxs, text


def get_bboxs_from_MS(json_path):
    """
    get_bboxs lee un archivo json entregado por OCR de Microsoft,
    extrae las frases y sus coordenadas en la imagen.

    argumentos:
        json_path (str) -- Ruta a json con las anotaciones
        
    return:
        bboxs (list(MOCR_BBox)) -- Bounding boxes.
        text (list(str)) -- frases reconocidas. 
    """
   
    if not path.exists(json_path):
        print("[Error] No existe el archivo", json_path)
        return None, None
    
    with open(json_path) as f:
        f_json = json.load(f)
    if f_json["succeeded"]:
        bboxs, text = zip(*[(MOCR_BBox(line["boundingBox"], 4), line["text"])  
                     for line in f_json['recognitionResult']["lines"]])
    else:
        print("[Info] Las anotaciones no fueron exitosas.")
        bboxs, text = None, None
    return bboxs, text
