"""
prepare_dataset.py se encarga de prepárar el dataset usando un template y sus anotaciones
"""
import argparse
from os import listdir, makedirs, path

import cv2
from tqdm import tqdm

from checkbox.align import align_to_template_tesseract
from checkbox.ocr import apply_ocr_tesseract, prepare_image, save_to_json
from checkbox.crop import get_checkboxes_from_template, save_checkboxes
from checkbox.classification import manual_binary_classify, split_test_data

# Argumentos de cmd
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to the template folder", type=str)
ap.add_argument("-i", "--images", required=True, help="Path to images folder", type=str)
ap.add_argument("-o", "--output", required=True, help="Path to output folder", type=str)
ap.add_argument("-g", "--height", required=False, help="Input images height", type=int, default=40)
ap.add_argument("-w", "--width", required=False, help="Input images width", type=int, default=40)
args = vars(ap.parse_args())

template_folder = args["template"]
images_folder = args["images"]
output_folder = args["output"]
width = args["width"]
height = args["height"]


if not path.exists(template_folder):
    print("[Error] No existe carpeta de template:", template_folder)
    exit(1)

if not path.exists(images_folder):
    print("[Error] No existe carpeta de imagenes:", images_folder)
    exit(1)

if path.exists(output_folder):
    print("[Error] Ya existe la carpeta de output:", output_folder)
    exit(1)

# Buscar los archivos del template
print("[Info] Buscando archivos de template")
template_bboxs_path = None
template_checkboxes_path = None
template_image_path = None
for file in listdir(template_folder):
    name, ext = path.splitext(file)
    if ext != ".json":
        template_image_path = path.join(template_folder, file)
    elif name == "checkboxes":
        template_checkboxes_path = path.join(template_folder, file)
    else:
        template_bboxs_path = path.join(template_folder, file)

if template_image_path is None or template_bboxs_path is None or template_checkboxes_path is None:
    print("[Error] Faltan archivos en la carpeta de template:", template_folder)
    exit(1)

# Preparar carpetas de salida
print("[Info] Preparando carpetas de salida")
images_bbox_folder = path.join(output_folder, "images", "annotations")
images_resized_folder = path.join(output_folder, "images", "resized")
images_aligned_folder = path.join(output_folder, "images", "aligned")
model_images_folder = path.join(output_folder, "model_data", "labeled", "train")
model_test_folder = path.join(output_folder, "model_data", "labeled", "test")
checkboxes_images_folder = path.join(output_folder, "model_data", "raw")


print("[Info] Creando", images_bbox_folder)
makedirs(images_bbox_folder)

print("[Info] Creando", images_resized_folder)
makedirs(images_resized_folder)

print("[Info] Creando", images_aligned_folder)
makedirs(images_aligned_folder)

print("[Info] Creando", model_images_folder)
makedirs(model_images_folder)

print("[Info] Creando", model_test_folder)
makedirs(model_test_folder)

print("[Info] Creando", checkboxes_images_folder)
makedirs(checkboxes_images_folder)


# Buscar las imagenes de entrenamiento
print("[Info] Buscando imagenes...", end="\t")
images_path = []
for img in listdir(images_folder):
    img_path = path.join(images_folder, img)
    images_path.append(img_path)

if len(images_path) == 0:
    print("[Error] No hay imagenes en", images_folder)
    exit(1)
print(f"{len(images_path)} imagenes encontradas")

print("[Info] Aplicando OCR a imagenes")
images = []
images_resized_path = []
images_bbox_path = []
for img_path in tqdm(images_path):
    # Resize imagenes
    img,_ = prepare_image(img_path)
    images.append(img)
    img_resized_path = path.join(images_resized_folder, path.basename(img_path))
    images_resized_path.append(img_resized_path)

    # Obtener bbox del ocr
    x,y,t = apply_ocr_tesseract(img)
    n, _ =  path.splitext(path.basename(img_path))
    img_bbox_path = path.join(images_bbox_folder, f"{n}.json")
    images_bbox_path.append(img_bbox_path)

    # Guardar resultados
    img.save(img_resized_path)
    save_to_json(x, y, t, img_bbox_path)


# Alinear imagenes de entrenamiento
print("[Info] Alineando imagenes")
images_aligned_path = []
for img_path, img_bbox_path in tqdm(list(zip(images_resized_path, images_bbox_path))):
    _, img_aligned = align_to_template_tesseract(template_image_path, img_path, template_bboxs_path, img_bbox_path)
    img_aligned_path = path.join(images_aligned_folder, path.basename(img_path))
    images_aligned_path.append(img_aligned_path)
    cv2.imwrite(img_aligned_path, img_aligned)


# Obtener bboxes de checkboxes
print("[Info] Recortando Checkboxes")
template_checkboxes = get_checkboxes_from_template(template_checkboxes_path)
for img_path in tqdm(images_aligned_path):
    # Recortar y obtenener imagens de entrenamiento
    save_checkboxes(img_path, template_checkboxes, folder=checkboxes_images_folder, resize=(height, width), gray=True)

# Clasificacion manuals
print("[Info] Comenzando clasificacion manual")
manual_binary_classify(checkboxes_images_folder, model_images_folder, "check", "not-check")

# Split data
print("[Info] Separando Test set")
split_test_data(model_images_folder, model_test_folder, 0.05)


print("[Info] Finalizado")