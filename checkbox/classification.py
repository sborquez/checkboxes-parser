import os
import shutil
from random import sample, shuffle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def manual_binary_classify(images_folder, output_folder, class_1, class_2, mv=True):
    """
    manual_binary_classify permite clasificar manualmente las imagen 
    contendias en 'images_folder' y las mueve a las carpetas correspondientes
    a su clase.

    Usar la flecha izq para clasificar como clase 1, la flecha der para la clase 2

    argumentos:
        images_folder (str) -- Ruta a la carpeta con imagenes
        output_folder (str) -- Ruta a la carpeta donde estaran las dos clases
        class_1 (str) -- Nombre de la clase 1
        class_2 (str) -- Nombre de la clase 2
        mv (bool) -- si es falso, las imagenes solo se copian
    
    return:
        None
    """ 
    
    if not os.path.exists(images_folder):
        print("[Error] No existe la carpeta de images", images_folder)

    # Crear carpetas de las clases
    os.makedirs(os.path.join(output_folder, class_1))
    os.makedirs(os.path.join(output_folder, class_2))

    cv2.namedWindow(f"[{class_1}] <- | -> [{class_2}]")
    for file_ in tqdm(os.listdir(images_folder)):
        file = os.path.join(images_folder, file_)
        img = cv2.imread(file)
        cv2.imshow(f"[{class_1}] <- | -> [{class_2}]", img)

        k = cv2.waitKey()

        if k == 81:
            #print("left")
            if mv:
                shutil.move(file, os.path.join(output_folder, class_1, file_))
            else:
                shutil.copyfile(file, os.path.join(output_folder, class_1, file_))
        elif k == 83:
            #print("right")
            if mv:
                shutil.move(file, os.path.join(output_folder, class_2, file_))
            else:
                shutil.copyfile(file, os.path.join(output_folder, class_2, file_))
        else:
            cv2.destroyAllWindows()
            return None


def split_test_data(train_data_folder, test_data_folder, test_portion=0.20):
    """
    split_test_data mueve una porcion de las imagenes de entrenamiento a una nueva
    carpeta para testing.

    argumentos:
        train_data_folder (str) -- Ruta a la carpeta con las imagenes de entrenamiento.
        test_data_folder (str) -- Ruta de salida a la carpeta de imagenes de testing.
        test_portion (float) -- (0.0-1.0) Porcion de imagenes que se moveran a testing.

    return:
        moved (int) -- Cantidad de imagenes movidas
    """
    if not os.path.exists(train_data_folder):
        print("[Error] No existe la carpeta de images", train_data_folder)

    # Crear carpetas de testing
    classes =  os.listdir(train_data_folder)
    train_folders = [os.path.join(train_data_folder, _class) for _class in classes]
    test_folders = [os.path.join(test_data_folder, _class) for _class in classes]
    for class_folder in test_folders:
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

    # Mover imagenes
    moved = 0
    for class_folder_train, class_folder_test in zip(train_folders, test_folders):
        imgs = os.listdir(class_folder_train)
        n_imgs = int(len(imgs) * test_portion)
        moved += n_imgs
        picked_imgs = sample(imgs, n_imgs)

        src_imgs = map(lambda train_img: os.path.join(class_folder_train, train_img), picked_imgs)
        dest_imgs = map(lambda test_img: os.path.join(class_folder_test, test_img), picked_imgs)
        for src, dest in zip(src_imgs, dest_imgs):
            print("[Info] Moving:", "..."+src[-50:], " -> ", "..."+dest[-50:])
            os.rename(src, dest)
    print("[Info] Imagenes en Test set:", moved)
    return moved


def build_model(width, height, classes, gray=True, model_path=None):
    """
    build_model contruye una red LeNet para la clasificacion de imagenes.

    argumentos:
        width (int) -- Ancho de las imagenes de entrada
        height (int) -- Altura de las imagenes de entrada
        classes (list(str)) -- Nombre de las clases a clasificar
        gray (bool) -- Si la imagenes vienen en esacalas de grises
        model_path (None|str) -- Si no es None, es la ruta de los pesos pre-entrenados 

    return:
        model (keras.models.Sequential) -- Modelo LeNet
        meta_data (dict) -- Informacion sobre el modelo
                            meta_data = {
                                "classes" : {
                                    0: <nombre_0>,
                                    ...
                                    n: <nombre_n>
                                },
                                "img_shape" : {
                                    "heigh" : <heigh>,
                                    "width" : <width>,
                                    "channels" : <channels>
                                },
                                "gray" : <bool>
                            }
    """

    meta_data = {
        "classes" : { i:name for i,name in enumerate(classes) },
        "img_shape" : {
            "height" : height,
            "width" : width,
            "channels" : 1 if gray else 3
        },
        "gray" : gray
    }

    if model_path is not None:
        model = load_model(model_path)

    else:
        if gray:
            input_shape = [height, width, 1]
        else:
            input_shape = [height, width, 3]
        
        if K.image_data_format() == "channels_first":
            input_shape = [input_shape[2]] + input_shape[:2]
        

        #Modelo secuencial
        model = Sequential()

        # Primera capa conv > relu > maxpool
        model.add(Conv2D(20, (5,5), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Segunda capa conv > relu > maxpool
        model.add(Conv2D(50, (5,5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Capa FC > relu
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # Capa softmax
        model.add(Dense(len(classes)))
        model.add(Activation("softmax"))

    return model, meta_data


def load_image(img_path, meta_data, expand=False):
    """
    load_image carga una imagen para ser usada como entreda del modelo

    argumentos:
        img_path (str) -- Ruta a la imagen
        meta_data (dict) -- Informacion sobre el modelo
        expand (bool) -- Si se usara esta imagen como unica entrada del modelo

    return:
        image (array) -- Imagen 
    """
    #print(img_path)
    image = cv2.imread(img_path)

    if image.shape != (meta_data["img_shape"]["height"], meta_data["img_shape"]["width"], 3):
        image = cv2.resize(image, (meta_data["img_shape"]["height"], meta_data["img_shape"]["width"]))

    if meta_data["gray"]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Se escalan las imagens a intensidades entre 0 y 1
    image = img_to_array(image) / 255.

    if expand:
        return np.expand_dims(image, axis=0)
    else:
        return image


def load_datasets(meta_data, train_data_folder, test_size=0.25):
    """
    load_datasets Carga las imagenes para el entrenamiento
    
    argumentos:
        meta_data (dict) -- Informacion sobre el modelo
        train_data_folder (str) -- Ruta a la carpeta con las imagenes de entrenamiento
        test_size (float) -- Proporcion de imagenes de set de validacion
    
    return:
        trainX, testX, trainY, testY (arrays, arrays, arrays, arrays) -- Entradas del modelo
    """
    # Cargar imagenes
    images_labels = []
    for label, name in meta_data["classes"].items():
        for img_name in os.listdir(os.path.join(train_data_folder, name)):
            # Cargar imagen
            image = load_image(os.path.join(train_data_folder, name, img_name), meta_data)

            # Se agregan a la lista junto a su label
            images_labels.append((image, label))

    # Se 'barajan' las clases
    shuffle(images_labels)

    # Se pasan a arrays
    images, labels = zip(*images_labels)
    X = np.array(images, dtype="float")
    y = np.array(labels)

    # Dividir el dataset en train y test (valid)
    (trainX, testX, trainY, testY) = train_test_split(X, y, test_size=test_size)

    # Convertir los labels a vectores
    trainY = to_categorical(trainY, num_classes=len(meta_data["classes"]))
    testY = to_categorical(testY, num_classes=len(meta_data["classes"]))

    return trainX, testX, trainY, testY


def load_test_dataset(meta_data, test_data_folder):
    """
    load_test_dataset Carga las imagenes del testset para la evaluacion
    
    argumentos:
        meta_data (dict) -- Informacion sobre el modelo
        test_data_folder (str) -- Ruta a la carpeta con las imagenes de testing
    
    return:
        X, Y (arrays, arrays) -- Entradas del modelo
    """
    # Cargar imagenes
    images_labels = []
    for label, name in meta_data["classes"].items():
        for img_name in os.listdir(os.path.join(test_data_folder, name)):
            # Cargar imagen
            image = load_image(os.path.join(test_data_folder, name, img_name), meta_data)

            # Se agregan a la lista junto a su label
            images_labels.append((image, label))

    # Se 'barajan' las clases
    shuffle(images_labels)

    # Se pasan a arrays
    images, labels = zip(*images_labels)
    X = np.array(images, dtype="float")
    y = np.array(labels)

    # Convertir los labels a vectores
    Y = to_categorical(y, num_classes=len(meta_data["classes"]))

    return X, Y


def train_model(model, meta_data, train_data_folder, output_model, epochs=100, init_lr=1e-3, batch_size=32, test_size=0.15, early=True, checkpoints=""):
    """
    train_model ajusta los parametros del modelos al set de entranamiento.
    
    argumentos:
        model (keras.models.Sequential) -- Modelo LeNet
        meta_data (dict) -- Informacion sobre el modelo
        train_data_folder (str) -- Ruta a la carpeta con las imagenes de entrenamiento
        output_model (str) -- Ruta donde se guardara el modelo
        epochs (int) -- Cantidad de epochs 
        int_lr (float) -- Learning rate inicial
        batch_size (int) -- Tamaño del batch
        test_size (float) -- Proporcion de imagenes de set de validacion
    
    return:
        H (History) -- Historial entregado por el entrenamiento del modelo
        output_model (str) -- Ruta donde se guarda el modelo
    """

    # Cargar imagenes
    print("[Info] Cargando datasets")
    trainX, testX, trainY, testY = load_datasets(meta_data, train_data_folder, test_size)
    print(f"[Info] Imagenes cargadas, train: {len(trainX)}\ttest: {len(testX)}")

    # Generador de imagenes, data augmentation
    image_generator = ImageDataGenerator(rotation_range=10, horizontal_flip=True,vertical_flip=True, fill_mode="nearest")

    # Inicializar el modelo
    print("[Info] Compilando modelo")
    model.compile(loss="binary_crossentropy", 
                  optimizer=Adam(lr=init_lr, decay=init_lr / epochs), 
                  metrics=["accuracy"])


    # Callbacks
    callbacks = []
    if early:
        callbacks.append(EarlyStopping(patience=10))

    if checkpoints != "":
        callbacks.append(ModelCheckpoint(checkpoints, period=5))

    # Entrenar el modelo
    print("[INFO] Entrenando modelo...")
    H = model.fit_generator(image_generator.flow(trainX, trainY, batch_size=batch_size),
	                        validation_data=(testX, testY), 
                            steps_per_epoch=len(trainX) // batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks)

    # save the model to disk
    print("[INFO] Guardando modelo...")
    model.save(output_model)                        
    return H, output_model


def evaluate_model(model, meta_data, test_data_folder):
    """
    evaluate_model obtiene el loss y accuracy del modelo utilizando las imagenes
    de test.

    argumentos:
        model (keras.models.Sequential) -- Modelo LeNet
        meta_data (dict) -- Informacion sobre el modelo
        test_data_folder (str) -- Ruta a la carpeta con las imagenes de testing

    return:
        None
    """
    print("[Info] Cargando test dataset")
    X, Y = load_test_dataset(meta_data, test_data_folder)
    print(f"[Info] Imagenes cargadas {len(X)}")

    loss, acc = model.evaluate(X, Y)
    print(f"[Info] loss: {loss}\taccuracy: {acc}")


def predict_image(model, meta_data, image_path):
    """
    predict_image realiza una prediccion a una imagen

    argumentos:
        model (keras.models.Sequential) -- Modelo LeNet
        meta_data (dict) -- Informacion sobre el modelo
        image_path (str) -- Ruta a la imagen

    return:
        class_name (str) -- Nombre de la clase
        score (float) -- Nivel de confianza
    """
    image = load_image(image_path, meta_data, True)
    result = model.predict(image)
    score = result.max()
    class_name = meta_data["classes"][result.argmax()]
    return class_name, score


def predict_images(model, meta_data, images_path):
    """
    predict_image realiza una prediccion a varias imagenes

    argumentos:
        model (keras.models.Sequential) -- Modelo LeNet
        meta_data (dict) -- Informacion sobre el modelo
        images_path (str) -- Ruta a la carpeta con imagenes
                    (list) -- Rutas a las imagenes

    return:
        result (list, list) -- lista de listas, contiene las clases
                                y las scores para cada imagen 
    """
    if isinstance(images_path, str):
        images = []
        for img in os.listdir(images_path):
            images.append(os.path.join(images_path, img))

    elif isinstance(images_path, list):
        images = images_path

    images = [load_image(img, meta_data) for img in images] 
    X = np.array(images)

    results = model.predict_on_batch(X)

    #print(results.argmax(axis=1).tolist())
    classes_names = [meta_data["classes"][i] for i in results.argmax(axis=1)]
    classes_score = results.max(axis=1).tolist()
    
    results = list(zip(classes_names, classes_score))
    #print(results)

    return results


def plot_train(H, save_plot_as):
    """
    plot_train Genera plot del entrenamiento

    argumentos:
        H (History) -- Historial entregado por el entrenamiento del modelo
        save_plot_as (str) -- Ruta donde guardar plot (termina en .png).
    return:
        None
    """
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = len(H.history["loss"])
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(save_plot_as)



if __name__ == "__main__":
    #manual_binary_classify("/home/ikari/Repo/checkboxes/checkbox_testdata/checkboxes", "/home/ikari/Repo/checkboxes/checkbox_testdata/model_train_data", "check", "no-check", False)
    #split_test_data("/home/ikari/Repo/checkboxes/checkbox_testdata/model_train_data", "/home/ikari/Repo/checkboxes/checkbox_testdata/model_test_data")
    #model, meta = build_model(40, 40, ["check", "no-check"])
    model, meta = build_model(40, 40,["check", "no-check"], True, "/home/ikari/Repo/checkboxes/script/test.model")

    #H, m = train_model(model, meta, "/home/ikari/Repo/checkboxes/checkbox_testdata/model_train_data", "test.model")
    #plot_train(H, "test.png")
    #evaluate_model(model, meta, "/home/ikari/Repo/checkboxes/checkbox_testdata/model_test_data")
    r=predict_images(model, meta, "/home/ikari/Repo/checkboxes/checkbox_testdata/model_test_data/check")    
