"""
prepare_template.py Prepara la imagen template de un formulario, gerera
sus checkboxes y puntos claves para la alineacion. 
"""

import argparse
import json
from os import makedirs, path

import cv2
import numpy as np

from checkbox.ocr import (apply_ocr_tesseract, prepare_image, save_image,
                          save_to_json)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image file", type=str)
ap.add_argument("-o", "--output", required=True, help="Path to output folder", type=str)
args = vars(ap.parse_args())

img_path = args["image"]
folder_path = args["output"]


if not path.exists(img_path):
    print("[Error] No existe la imagen:", img_path)

if not path.exists(folder_path):
    print("[Info] Creando directorio:",folder_path)
    makedirs(folder_path)


image_filename = path.basename(img_path)
image_name, image_ext = path.splitext(image_filename)
output_image_filepath = path.join(folder_path, image_filename)

json_filename = f"{image_name}.json"
json_path = path.join(folder_path, json_filename)
checkbox_path = path.join(folder_path, "checkboxes.json")

# Resize imagen a tamaño valido
print("[Info] Preparando imagen")
image, ratio = prepare_image(img_path)
save_image(image, output_image_filepath)

# Obtener bbox del ocr
print("[Info] Aplicando OCR")
x, y, text = apply_ocr_tesseract(image)
save_to_json(x, y, text, json_path)

# Obtener bbox de los checkboxes
print("[Info] Marcar Checkboxes")
aux_rect = []
mouse_pos = []
pos_rects = []
neg_rects = []
names_rects = []
cropping = False
positive = True

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global pos_rects, neg_rects, aux_rect, positive, mouse_pos, cropping, labeling
    if labeling:
        return None
 
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        aux_rect = (x, y)
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        if positive:
            pos_rects.append((aux_rect, (x, y)))
        else:
            neg_rects.append((aux_rect, (x, y)))
        aux_rect = []
        positive = not positive
        cropping = False

    mouse_pos = (x, y)


W, H = image.size
r = 750 / H
image = image.resize((int(W*r), int(H*r)))
image = np.array(image)

cv2.namedWindow("[u] undo - [c]  cancel - [f] finish (start labeling)")
cv2.setMouseCallback("[u] undo - [c]  cancel - [f] finish (start labeling)", click_and_crop)

print("[Info] Comienza labeling")
labeling = False
while True:
    clone = np.copy(image)
	# display the image and wait for a keypress
    for rect in pos_rects:
        cv2.rectangle(clone, rect[0], rect[1], (0, 255, 0), 1)
    for rect in neg_rects:
        cv2.rectangle(clone, rect[0], rect[1], (0, 0, 255), 1)
        
    if cropping:
        cv2.rectangle(clone, aux_rect, mouse_pos, (125, 125, 125), 1)
        
    cv2.imshow("[u] undo - [c]  cancel - [f] finish (start labeling)", clone)
    

    key = cv2.waitKey(1) & 0xFF
    # if the 'u' key is pressed, undo last action
    if key == ord("u"):
        if len(pos_rects) == len(neg_rects):
            neg_rects=neg_rects[:-1]
            # names_rects=names_rects[:-1]
            positive = False
        else:
            pos_rects=pos_rects[:-1]
            positive = True

    # if the 'f' key is pressed, break from the loop and start labeling    
    if key == ord("f") and len(pos_rects) == len(neg_rects):
        labeling = True
        break
	
    # if the 'c' key is pressed, break from the loop and cancel
    if key == ord("c"):
        # close all open windows
        cv2.destroyAllWindows()
        exit(1)

 # start labeling
items = []
total = len(pos_rects)
while labeling:
    clone = np.copy(image)
	# display the image and wait for a keypress
    if len(pos_rects) == 0:
        labeling = False
        continue

    if len(pos_rects) > 0:
        pos = pos_rects.pop(0)
        neg = neg_rects.pop(0)
   
    #TODO mostrar el rectangulo
    #cv2.rectangle(clone, pos[0], neg[1], (0, 0, 255), 5)
    
    cv2.imshow("[u] undo - [c]  cancel - [f] finish (start labeling)", image)
    new_item = {
        "nombre": input("Ingrese nombre: "),
        "boxneg": list(map(lambda p: (int(p[0]/r), int(p[1]/r)), neg)),
        "boxpos": list(map(lambda p: (int(p[0]/r), int(p[1]/r)), pos))
    }
    items.append(new_item)

# close all open windows
cv2.destroyAllWindows()


print("[Info] Guardando anotaciones")
# standarize boxes, top left corner, bottom right 
for i in items:
    boxneg = i["boxneg"]
    x1 = min(boxneg[0][0], boxneg[1][0])
    y1 = min(boxneg[0][1], boxneg[1][1])
    x2 = max(boxneg[0][0], boxneg[1][0])
    y2 = max(boxneg[0][1], boxneg[1][1])
    i["boxneg"]= [[x1,y1],[x2,y2]]

    boxpos = i["boxpos"]
    x1 = min(boxpos[0][0], boxpos[1][0])
    y1 = min(boxpos[0][1], boxpos[1][1])
    x2 = max(boxpos[0][0], boxpos[1][0])
    y2 = max(boxpos[0][1], boxpos[1][1])
    i["boxpos"] = [[x1,y1],[x2,y2]]


# Save annotations
with open(checkbox_path, "w") as f:
    json.dump(items, f)


print("[Info] Resultados en:", folder_path)
