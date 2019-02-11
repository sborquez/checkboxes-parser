"""
evaluate_models.py Calcula varias metricas a diferentes modelos utilizandos varios datasets
"""

import argparse
import pprint
from os import listdir, path, makedirs
from shutil import copy
import numpy as np
from checkbox.classification import build_model, load_image
import json
from prettytable import PrettyTable

# Argumentos de cmd
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--datasets", required=True, help="List of paths to the datasets (test folder)", type=str, nargs="+")
ap.add_argument("-m", "--model", required=True, help="Path to model", type=str)
ap.add_argument("-o", "--output", required=False, help="Path to output folder", type=str, default=".")
ap.add_argument("-g", "--height", required=False, help="Input images height", type=int, default=40)
ap.add_argument("-w", "--width", required=False, help="Input images width", type=int, default=40)
ap.add_argument("-p", "--precision", help="Calculate precision", dest="precision", action="store_true")
ap.add_argument("-r", "--recall", help="Calculate recall", dest="recall", action="store_true")
ap.add_argument("-f", "--f1score", help="Calculate f1 score", dest="f1score", action="store_true")
ap.add_argument("-j", "--json", help="Save json file", dest="save_json", action="store_true")
ap.set_defaults(precision=False, recall=False, f1score=False, save_json=False)

args = vars(ap.parse_args())

datasets_paths = args["datasets"]
model_path = args["model"]
output_folder = path.join(args["output"], path.splitext(path.basename(model_path))[0])
height = args["height"]
width = args["width"]
precision = args["precision"]
recall = args["recall"]
f1score = args["f1score"]
save_json = args["save_json"]

for ds in datasets_paths:
    if not path.exists(ds):
        print("[Error] No existe el dataset", ds)
        exit(1)

if not path.exists(model_path):
    print("[Error] No existe el modelo", model_path)
    exit(1)

if not path.exists(output_folder):
    makedirs(output_folder)


# Instanciar modelo
print("[Info] Instanciando modelo")
model, meta_data = build_model(width, height, ["check", "not-check"], gray=True, model_path=model_path)


datasets_results = {}
for ds in datasets_paths:
    # Preparar resultados del dataset
    datasets_forms = {}
    images_path = []
    images_outputs = []
    images_form = []
    wrong_images = []

    # Preparar imagenes
    for f in listdir(path.join(ds, "check")):
        images_path.append(path.join(ds, "check", f))
        images_outputs.append("check")
        form = path.basename(f).split(":")[0]
        if form not in datasets_forms:
            datasets_forms[form] = {"check":0, "not-check":0, "predicted_check": 0, "predicted_not-check":0}
        datasets_forms[form]["check"] += 1
        images_form.append(form)

    for f in listdir(path.join(ds, "not-check")):
        images_path.append(path.join(ds, "not-check", f))
        images_outputs.append("not-check")
        form = path.basename(f).split(":")[0]
        if form not in datasets_forms:
            datasets_forms[form] = {"check":0, "not-check":0, "predicted_check": 0, "predicted_not-check":0}
        datasets_forms[form]["not-check"] += 1
        images_form.append(form)

    # Cargar imagenes
    images = [load_image(img, meta_data) for img in images_path] 
    X = np.array(images)

    # Predict
    results = model.predict_on_batch(X).argmax(axis=1)
    results = [meta_data["classes"][r] for r in results]

    # Evaluate
    TP = TN = FP = FN = 0
    for gt, r, img in zip(images_outputs, results, images_path):
        form = path.basename(img).split(":")[0]
        if r == "check":
            datasets_forms[form]["predicted_check"] += 1
            if r == gt:
                TP += 1
            else:
                FP += 1
                wrong_images.append(img)
        elif r == "not-check":
            datasets_forms[form]["predicted_not-check"] += 1
            if r == gt:
                TN += 1
            else:
                FN += 1       
                wrong_images.append(img)

    acc = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) != 0 else 0
    prec = TP /(TP + FP) if (TP + FP) != 0 else 0
    rec = TP / (TP + FN) if (TP+FN) != 0 else 0
    f1 = 2*(prec*rec)/(prec+rec) if (prec+rec) != 0 else 0

    
    ds_result = {
        "Accuracy": acc,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Misslabels" : wrong_images,
        "forms" : datasets_forms
    }

    makedirs(path.join(output_folder, "misslabels", path.basename(ds)))
    for src_img in wrong_images:
        dst_img =  path.join(output_folder, "misslabels", path.basename(ds), path.basename(src_img))
        copy(src_img, dst_img)

    if precision:
        ds_result["Precision"] = prec
    if recall:
        ds_result["Recall"] = rec
    if f1score:
        ds_result["F1 Score"] = f1

    datasets_results[path.basename(ds)] = ds_result

if save_json:
    out_json = path.join(output_folder, path.splitext(path.basename(model_path))[0] + "_result.json")
    with open(out_json, "w") as f:
        json.dump(datasets_results, f)

pp = pprint.PrettyPrinter(indent=2)
#pp.pprint(datasets_results)

header = ["Dataset", "Accuracy", "TP", "TN", "FP", "FN", "forms"]
if precision:
    header.append("Precision")
if recall:
    header.append("Recall")
if f1score:
    header.append("F1 Score")

t = PrettyTable(header)
for ds, scores in datasets_results.items():
    row = [ds]
    for score in header[1:]:
        if score == "forms":
            corrects = 0
            total = len(scores[score])
            for form in scores[score].values():
                if form["check"] == form["predicted_check"] and form["not-check"] == form["predicted_not-check"] :
                    corrects += 1
            row.append(f"{corrects}/{total}")
        else:
            row.append(round(scores[score], 3))
    t.add_row(row)
            
with open(path.join(output_folder,"scores.txt"),"w") as r:
    r.write(t.get_string())