"""
generate_model.py  Crea y entrena un modelo a partir de un dataset generado anteriormente
"""

import argparse
import uuid
from os import listdir, makedirs, path

from checkbox.classification import (build_model, evaluate_model, plot_train,
                                     train_model)

# Argumentos de cmd
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Path to images folder", type=str)
ap.add_argument("-o", "--output", required=True, help="Path to output folder", type=str)
ap.add_argument("-g", "--height", required=False, help="Input images height", type=int, default=40)
ap.add_argument("-w", "--width", required=False, help="Input images width", type=int, default=40)
ap.add_argument("-n", "--name", required=False, help="Model name", type=str, default="")
ap.add_argument("-m", "--model", required=False, help="Path to pre-trainned model", type=str, default="")
ap.add_argument("-e", "--early", help="Early stop", dest="early", action="store_true")
ap.add_argument("-c", "--checkpoints", help="Checkpoints", dest="checkpoints", action="store_true")
ap.set_defaults(early=False, checkpoints=False)
args = vars(ap.parse_args())

datasets_folder = args["images"]
output_folder = args["output"]
width = args["width"]
height = args["height"]
model_path = args["model"] if args["model"] != "" else None 
model_name = args["name"] if args["name"] != "" else uuid.uuid4().hex[:10] + ".hdf5"

# Creando carpetas
train_data_folder = path.join(datasets_folder, "train")
test_data_folder = path.join(datasets_folder, "test")
checkpoints_folder = path.join(output_folder, "checkpoints")
output_model = path.join(output_folder, model_name)

if not path.exists(train_data_folder):
    print("[Error] No existe la carpeta de imagenes", train_data_folder)
    exit(1)

if not path.exists(test_data_folder):
    print("[Error] No existe la carpeta de imagenes", test_data_folder)
    exit(1)

if not path.exists(checkpoints_folder):
    makedirs(checkpoints_folder)

# Callbacks
early = args["early"]
checkpoints = path.join(checkpoints_folder, model_name+".{epoch:02d}-{val_loss:.2f}.hdf5") if args["checkpoints"] else ""

# Construccion del modelo
model, metadata = build_model(width, height,["check", "not-check"], True, model_path)

# Entrenamiento
hist, _ = train_model(model, metadata, train_data_folder, output_model, epochs=100, 
                    init_lr=1e-4, batch_size=64, test_size=0.10, checkpoints=checkpoints, early=early)

# Guardar resultados
plot_train(hist, output_model + ".png")
evaluate_model(model, metadata, test_data_folder)
