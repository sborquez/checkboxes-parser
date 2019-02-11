"""
mask.py modulo

Coleccion de funciones para la aplicacion de mascara de un documento template
sobre imagenes de documentos alineados.
"""

import json
from os import makedirs, path

import cv2
import matplotlib.pyplot as plt

from .shapes import *

PAD = 5


def get_checkboxes_from_template(checkboxes_path):
    """
    get_checkboxes_from_template genera un diccionario a partir de las 
    anotaciones en el archivo json del template.
    
    argumentos:
        checkboxes_path (str) -- Ruta al archivo .json con la anotaciones de 
                                 los checkboxes del template
    
    return:
        checkboxes (dict) -- Diccionario con las coordenadas de los checkboxes 
                             del template.
                             checkboxes = {
                                ...
                                "<id>" : {
                                    "S" : [(<x_start>, <y_start>), 
                                        (<x_end>,   <y_end>)],
                                    "N" : [(<x_start>, <y_start>), 
                                        (<x_end>,   <y_end>)],
                                },
                                ...
                             }
    """
    if not path.exists(checkboxes_path):
        print("[Error] No existe el archivo", checkboxes_path)
        return None

    with open(checkboxes_path) as f:
        data = json.load(f)
    
    checkboxes = {}
    try:
        for obj in data:
            checkboxes[obj["nombre"]] = {
                "S" : obj["boxpos"], 
                "N" : obj["boxneg"]
            }
    except Exception as err:
        print("Ocurio un error con los checkboxes", err)
        return None

    return checkboxes


def get_boxes_from_image(image_path, checkboxes, resize=None, gray=True, pad=PAD):
    """
    get_boxes_from_image recorta los checkboxes de una imagen alineada a partir
    de los checkboxes generados con el template.

    argumentos:
        image_path (str)  -- Ruta a la imagen alineada 
        checkboxes (dict) -- Diccionario construido a partir del template que
                             contiene las posisciones de los checkboxes.
        resize (None | Tuple(H, W)) -- Si no es None, los recortes se redimensionan a
                                       a los valores de resize.
        gray (bool) -- Usar escala de grises
    
    return:
        image_checkboxes (dict) -- Contiene los recortes de los checkboxes de la imagen
                                   correspondientes al template
    """

    image = cv2.imread(image_path)
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image_checkboxes = {}
    for item_id, checkboxes in checkboxes.items():
        # cortar checkbox positivo
        (x_s, y_s), (x_e, y_e) = checkboxes["S"]
        #positive = np.copy(image[x_s:x_e, y_s:y_e])
        positive = np.copy(image[y_s-pad:y_e+pad, x_s-pad:x_e+pad])

        # cortar checkbox negativo
        (x_s, y_s), (x_e, y_e) = checkboxes["N"]
        #negative = np.copy(image[x_s:x_e, y_s:y_e])
        negative = np.copy(image[y_s-pad:y_e+pad, x_s-pad:x_e+pad])

        if isinstance(resize, tuple) and len(resize) == 2:
            positive = cv2.resize(positive, resize)
            negative = cv2.resize(negative, resize)

        image_checkboxes[item_id] = {
            "S":positive,
            "N":negative
        }

    return image_checkboxes


def show_projection(image_path, checkboxes):
    """
    show_projection muestra donde se ubican los checkbox del template en la 
    imagen.

    argumentos:
        image_path (str) -- Ruta a la imagen
        checkboxes (dict) -- Diccionario construido a partir del template que
                             contiene las posisciones de los checkboxes.
    return:
        None
    """
    plt.figure(figsize=(20,20))
    image = cv2.imread(image_path)
    plt.imshow(image)

    for chbox in checkboxes.values():
        (x_s, y_s), (x_e, y_e) = chbox["S"]
        plt.scatter([x_s, x_e], [y_s, y_e])

        (x_s, y_s), (x_e, y_e) = chbox["N"]
        plt.scatter([x_s, x_e], [y_s, y_e])

    plt.show()


def save_checkboxes(image_path, checkboxes, folder=".", resize=None, gray=True, pad=PAD):
    """
    save_checkboxes guarda los recortes de los checkbox de una imagen en 'folder'.

    argumentos:
        image_path (str)  -- Ruta a la imagen alineada 
        checkboxes (dict) -- Diccionario construido a partir del template que
                             contiene las posisciones de los checkboxes.
        folder (str) --- Ruta a una carpeta donde se guardaran los recortes, si
                         la carpeta no existe, se crea.
        resize (None | Tuple(H, W)) -- Si no es None, los recortes se 
                                       redimensionan a los valores de resize.
        gray (bool) -- Usar escala de grises
    
    return:
        None
    """

    checkboxes = get_boxes_from_image(image_path, checkboxes, resize, gray, pad)
    if not path.exists(folder):
        makedirs(folder)
    image_name = path.basename(image_path)
    image_name, image_ext = path.splitext(image_name)
    for item_id, checkboxes in checkboxes.items():
        cbp_name = f"{image_name}_:POS:{item_id}{image_ext}"
        cbn_name = f"{image_name}_:NEG:{item_id}{image_ext}"
        cv2.imwrite(path.join(folder, cbp_name), checkboxes["S"])
        cv2.imwrite(path.join(folder, cbn_name), checkboxes["N"])


def show_checkboxes(image_path, checkboxes, resize=None, gray=True, pad=PAD):
    """
    show_checkboxes pĺotea los recortes de los checkbox de una imagen.

    argumentos:
        image_path (str)  -- Ruta a la imagen alineada 
        checkboxes (dict) -- Diccionario construido a partir del template que
                             contiene las posisciones de los checkboxes.
        resize (None | Tuple(H, W)) -- Si no es None, los recortes se 
                                       redimensionan los valores de resize.
        gray (bool) -- Usar escala de grises
    
    return:
        None
    """


    checkboxes = get_boxes_from_image(image_path, checkboxes, resize, gray, pad)
    for item_id, checkboxes in checkboxes.items():
        plt.figure(figsize=(20,10))
        if gray:
            plt.subplot(121)
            plt.imshow(checkboxes["S"], cmap="gray",vmin=0, vmax=255)
            plt.title("Positive")

            plt.subplot(122)
            plt.imshow(checkboxes["N"], cmap="gray",vmin=0, vmax=255)
            plt.title("Negative")
        else:
            plt.subplot(121)
            plt.imshow(checkboxes["S"][:,:,::-1],vmin=0, vmax=255)
            plt.title("Positive")

            plt.subplot(122)
            plt.imshow(checkboxes["N"][:,:,::-1],vmin=0, vmax=255)
            plt.title("Negative")

        plt.suptitle(f"{path.basename(image_path)} {item_id}")
        plt.show()

if __name__ == "__main__":
    from os import listdir
    chbxs = get_checkboxes_from_template("/home/ikari/Repo/checkboxes/checkbox_testdata/checkboxes.json")
    folder = "/home/ikari/Repo/checkboxes/checkbox_testdata/aligned/"
    #folder = "/home/ikari/Repo/checkboxes/checkbox_testdata/resized/"
    for file in listdir(folder):
        if ".json" in file:
            continue
        save_checkboxes(path.join(folder, file), chbxs, folder="outputs", resize=(40,40), gray=True)
        #show_projection(path.join(folder, file), chbxs)
        #show_checkboxes(path.join(folder, file), chbxs, gray=True, resize=(40,40))
