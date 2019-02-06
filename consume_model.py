"""
consume_model.py  instancia un modelo entrenado y obtiene el resultado de 
analizar una o varias imagenes de un tipo de formulario.
"""
import argparse
import json
import cv2
from shutil import rmtree
from os import listdir, makedirs, path
from checkbox.classification import build_model, predict_images
from checkbox.align import align_to_template_tesseract
from checkbox.ocr import apply_ocr_tesseract, prepare_image, save_to_json, save_image
from checkbox.crop import get_checkboxes_from_template, save_checkboxes


temp_folder = "./tmp"

# Argumentos de cmd
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to the template folder", type=str)
ap.add_argument("-i", "--image", required=True, help="Path to image", type=str)
ap.add_argument("-m", "--model", required=True, help="Path to model", type=str)
ap.add_argument("-o", "--output", required=False, help="Path to output folder", type=str, default=".")
ap.add_argument("-g", "--height", required=False, help="Input images height", type=int, default=40)
ap.add_argument("-w", "--width", required=False, help="Input images width", type=int, default=40)
ap.add_argument("-f", "--full", help="Show complete result", dest="full", action="store_true" )
ap.set_defaults(full=False)
args = vars(ap.parse_args())

template_folder = args["template"]
model_path = args["model"]
width = args["width"]
height = args["height"]
output_folder = args["output"]
image_path = args["image"]
image_name = path.basename(image_path)

results_path = path.join(output_folder, path.splitext(image_name)[0] + ".json")
full = args["full"]

if not path.exists(template_folder):
    print("[Error] No existe carpeta de template:", template_folder)
    exit(1)

if not path.exists(image_path):
    print("[Error] No existe la imagen:", image_path)
    exit(1)

if not path.exists(model_path):
    print("[Error] No existe el modelo:", model_path)
    exit(1)

if not path.exists(output_folder):
    makedirs(output_folder)

if not path.exists(temp_folder):
    makedirs(temp_folder)


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

# Resize imagen
print("[Info] Preparando imagen")
image,_ = prepare_image(image_path)
temp_image_path = path.join(temp_folder, image_name) 

# Obtener bbox del ocr
print("[Info] Aplicando OCR")
x,y,t = apply_ocr_tesseract(image)
temp_bboxs_path = path.join(temp_folder, image_name+".json")

# save temp files
print("[Info] Guardando archivos temporales")
save_image(image, temp_image_path)
save_to_json(x, y, t, temp_bboxs_path)

# align
print("[Info] Alinieando imagen")
success, image_aligned = align_to_template_tesseract(template_image_path, temp_image_path, template_bboxs_path, temp_bboxs_path)
if not success:
    print("[Error] No se pudo alinear imagen")    
    exit(1)
cv2.imwrite(temp_image_path, image_aligned)

# Obtener bboxes de checkboxes
print("[Info] Recortando Checkboxes")
template_checkboxes = get_checkboxes_from_template(template_checkboxes_path)

# Recortar y obtenener imagenes a evaluar
checkboxes_images_folder = path.join(temp_folder, "checkboxes")
makedirs(checkboxes_images_folder)
save_checkboxes(temp_image_path, template_checkboxes, 
                folder=checkboxes_images_folder, resize=(height, width), gray=True)

# Instanciar modelo
print("[Info] Instanciando modelo")
model, meta_data = build_model(width, height, ["check", "not-check"], gray=True, model_path=model_path)

# classify
print("[Info] Realizando predicciones")
images_path = [path.join(checkboxes_images_folder, img_path) for img_path in listdir(checkboxes_images_folder)]
predictions = predict_images(model, meta_data, images_path)

# Guardando resultados
results = {}
for checkbox, prediction in zip(images_path, predictions):
    checkbox = path.splitext(path.basename(checkbox))[0]
    _checkbox =  checkbox.split(":")
    if len(_checkbox) != 3:
        print("[Error] Formato incorrecto", checkbox)
        exit(1)
    _,option,item = _checkbox # imagen:[POS|NEG]:item_id
    if item not in results:
        results[item] = {}
    results[item][option] = prediction[0]


# filtrar y validar resultados
if not full:
    filtered_results = {}
    for item, checks in results.items():
        if checks["POS"] == checks["NEG"]:
            filtered_results[item] = "invalido"
        elif checks["POS"] == "check":
            filtered_results[item] = "POS"
        else:
            filtered_results[item] = "NEG"
    results = filtered_results

# Guardar resultados
print("[Info] Guardando resultados")
print(results)
with open(results_path, "w") as f:
    json.dump(results, f)

# Borrar archivos temporales
print("[Info] Borrando archivos temporales")
rmtree(temp_folder)
