"""
align.py modulo

Coleccion de funciones para la alineacion de dos imagenes de documentos.

"""



import json
from collections import Counter
from math import ceil
from os import path
from random import sample

import cv2
from skimage.measure import compare_ssim

from .ocr import *
from .shapes import *

#from pytesseract import image_to_osd, Output

try:
    from PIL import Image
except ImportError:
    import Imagetqdm


def match(bboxs1, text1, bboxs2, text2, MS=True):
    """
    match filtra las frases detectadas que se encuentren en ambos documentos.

    argumentos:
        bboxs1 (list(MOCR_BBox)) -- Bounding boxes de la primera images.
        text1 (list(str)) -- frases reconocidas de la primera images. 
        bboxs2 (list(MOCR_BBox)) -- Bounding boxes de la segunda images.
        text2 (list(str)) -- frases reconocidas de la segunda images. 

        centroid (bool) -- Utilizar solo el centro de las bounding boxes.

    return:
        pairs (list((tuple, tuple))) -- lista de pares de coordenadas de las
        palabras que se repiten en ambas imaágenes.
    """
    # La palabras claves son las que aparecen solo una vez en cada documento.
    pairs = []
    if MS:
        counts1 = filter(lambda k: k[1]==1, Counter(text1).items())
        counts2 = filter(lambda k: k[1]==1, Counter(text2).items())
        
        matched_lines = set(counts1) & set(counts2)
        for m,_ in matched_lines:
            b1 = bboxs1[text1.index(m)]
            b2 = bboxs2[text2.index(m)]
            pairs.append((b1.get_centroid(), b2.get_centroid()))
            
    else:
        matched_lines = set(text1) & set(text2)
        for m in matched_lines:
            b1 = bboxs1[text1.index(m)]
            b2 = bboxs2[text2.index(m)]
            pairs.append((b1.p1.get_tuple(), b2.p1.get_tuple()))
          
    return pairs


def transform(img1, img2, matched_points):
    """
    transform calcula la matriz de homografia entre las dos imagenes y transforma
    la img2 a img1 usandos los puntos claves, matched_points.

    argumentos:
        img1 (array) -- Imagen base.
        img2 (array) -- Imagen a transformar.

    return:
        warped_img2 (array) -- img2 transformada.

    """
    p1, p2 = zip(*matched_points)
    points1 = np.array(p1, dtype=np.float32)
    points2 = np.array(p2, dtype=np.float32)

    H, _ = cv2.findHomography(points2, points1, cv2.RANSAC)    
    #print(H)
    
    height, width = img1.shape
    warped_img2 = cv2.warpPerspective(img2, H, (width, height))
    return warped_img2


def align_to_template_tesseract(template_img_path, source_img_path, template_json_path, source_json_path):
    """
    align_to_template alinea la imagen source al template.

    argumentos:
        template_img_path (str) -- Ruta a la imagen template.
        source_img_path (str) -- Ruta a la imagen source.
        template_json_path (str) -- Ruta a json con anotaciones del template
        source_json_path (str) -- Ruta a json con anotaciones de la imagen


    return:
        success (bool) -- Se pudo realizar la transformacion.
        result_image (array) -- Imagen alineada.
    """
    if not path.exists(template_img_path):
        print("[Error] No existe el archivo", template_img_path)
        return False, None

    if not path.exists(source_img_path):
        print("[Error] No existe el archivo", source_img_path)
        return False, None

    if not path.exists(template_json_path):
        print("[Error] No existe el archivo", template_json_path)
        return False, None

    if not path.exists(source_json_path):
        print("[Error] No existe el archivo", source_json_path)
        return False, None

    try:
        template = cv2.imread(template_img_path)
        source = cv2.imread(source_img_path)
        
        template =  cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        source =  cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    except Exception as err:
        print("[Error] Ocurrio un error cargando las imagenes", err)
        return False, None


    templ_bboxs, templ_text = get_bboxs_from_tesseract(template_json_path)
    src_bboxs, src_text = get_bboxs_from_tesseract(source_json_path)

    if templ_bboxs is None or src_bboxs is None:
        print("[Info] No se identificaron bounding boxes")
        return False, None


    matched_points = match(templ_bboxs, templ_text, src_bboxs, src_text, False)
    if len(matched_points) < 4:
        print("[Info] No hay suficientes puntos claves")
        return False, None

    result_image = transform(template, source, matched_points)
    return True, result_image


def align_to_template_MS(template_img_path, source_img_path, template_json_path, source_json_path):
    """
    align_to_template alinea la imagen source al template.

    argumentos:
        template_img_path (str) -- Ruta a la imagen template.
        source_img_path (str) -- Ruta a la imagen source.
        template_json_path (str) -- Ruta a json con anotaciones del template
        source_json_path (str) -- Ruta a json con anotaciones de la imagen


    return:
        success (bool) -- Se pudo realizar la transformacion.
        result_image (array) -- Imagen alineada.
    """
    if not path.exists(template_img_path):
        print("[Error] No existe el archivo", template_img_path)
        return False, None

    if not path.exists(source_img_path):
        print("[Error] No existe el archivo", source_img_path)
        return False, None

    if not path.exists(template_json_path):
        print("[Error] No existe el archivo", template_json_path)
        return False, None

    if not path.exists(source_json_path):
        print("[Error] No existe el archivo", source_json_path)
        return False, None


    try:
        template = cv2.imread(template_img_path)
        source = cv2.imread(source_img_path)
        
        template =  cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        source =  cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    except Exception as err:
        print("[Error] Ocurrio un error cargando las imagenes", err)
        return False, None

    templ_bboxs, templ_text = get_bboxs_from_MS(template_json_path)
    src_bboxs, src_text = get_bboxs_from_MS(source_json_path)

    if templ_bboxs is None or src_bboxs is None:
        print("[Info] No se identificaron bounding boxes")
        return False, None


    matched_points = match(templ_bboxs, templ_text, src_bboxs, src_text)
    if len(matched_points) < 4:
        print("[Info] No hay suficientes puntos claves")
        return False, None

    result_image = transform(template, source, matched_points)
    return True, result_image


# def robust_align_to_template(template_img_path, source_img_path, ratio, retries=10):
#     raise NotImplementedError
#     """
#     robust_align_to_template alinea la imagen source al template el mejor resultado
#     al elegir un porcion de los puntos.

#     argumentos:
#         template_img_path (str) -- Ruta a la imagen template.
#         source_img_path (str) -- Ruta a la imagen source.
#         ratio (float) [0-1] -- Porcion de puntos seleccionados.
#         retries (int) -- Cantidad de reintentos para elegir los puntos.

#     return:
#         success (bool) -- Se pudo realizar la transformacion.
#         best_result_image (array) -- Imagen alineada.
#         best_result_score (float) -- Puntaje alcanzado
#     """
#     if not path.exists(template_img_path):
#         print("[Error] No existe el archivo", template_img_path)
#         return False, None, 0.0

#     if not path.exists(source_img_path):
#         print("[Error] No existe el archivo", source_img_path)
#         return False, None, 0.0

#     try:
#         template = cv2.imread(template_img_path)
#         source = cv2.imread(source_img_path)
        
#         template =  cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
#         source =  cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
#     except Exception as err:
#         print("[Error] Ocurrio un error cargando las imagenes", err)
#         return False, None, 0.0

#     templ_bboxs, templ_text = get_bboxs(template_img_path)
#     src_bboxs, src_text = get_bboxs(source_img_path)

#     if templ_bboxs is None or src_bboxs is None:
#         print("[Info] No se identificaron bounding boxes")
#         return False, None, 0.0


#     matched_points = match(templ_bboxs, templ_text, src_bboxs, src_text)
#     if len(matched_points) < 4:
#         print("[Info] No hay suficientes puntos claves")
#         return False, None, 0.0

#     pick = ceil(len(matched_points)*ratio)
#     pick = pick if pick >= 4 else 4

#     best_result_image = None
#     best_result_score = 0.0
#     while retries:
#         picked_matched_points = sample(matched_points, pick)
#         result_image = transform(template, source, picked_matched_points)
#         result_score = compare_ssim(template, result_image)
#         if result_score > best_result_score:
#             best_result_image = result_image
#             best_result_score = result_score

#         print("score", result_score, "\tbest",best_result_score)
#         retries -= 1

#     return True, best_result_image, best_result_score
