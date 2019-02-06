# Checkboxes

Este respositorio contiene un conjunto de módulos y scripts para la automatización en la revisión de formularios con checkboxes.

Esta herramienta permite la creación de un dataset, entrenamiento del modelo de clasificación y el consumo de este modelo.

## Estructura

### Módulos
* __ocr.py__ Aplicación de OCR-tesseract sobre imágenes de documentos.
* __align.py__ Alineación de imagenes de documentos utilizando OCR.
* __crop.py__ Recorte de imágenes de checkboxes de un documento.
* __shapes.py__ Bounding Boxes.
* __classification.py__ Clasificación manual y automatica de imágenes.


### Scripts
* __prepare_template.py__ Prepara una imagen para ser utilizada como template de formulario. Obtiene una imagen redimensionada, sus *puntos claves* (KP) para la alineación y posiciones de los *checkbboxes*.
     * __-i__:  Ruta a una imagen que se quiere utilizar como template.
     * __-o__:  Ruta a una carpera que el script creará para guardar los resultados. 


```
$ python prepare_template.py -i <ruta/a/imagen/template> -o <ruta/a/carpeta/de/salida>
```

* __prepare_dataset.py__ Genera un dataset de entrenamiento y testing para un modelo de clasificación. A partir de una imagen template, obtiene los checkboxes de nuevas imagenes, luego se clasifican manualmente y se separan train/test sets. Como resultado se obtienen las imágenes redimencionadas, alineadas y los datasets train/test. 
    * __-t__: Ruta a la carpeta que contiene los datos del template, generada por _prepare_template.py__.
    * __-i__: Ruta a la carpeta contenedora de las imagenes de formularios, a partir de estas imagenes se generará el dataset.
    * __-o__: Ruta a una carpeta que el script creará, en esta se guardarán los resultados del script.
    * __-g__ (opcional): Altura de las imagenes del dataset. Como default se usa 40px.
    * __-w__ (opcional): Ancho de las imagenes del dataset. Como default se usa 40px.
```
$ prepare_dataset.py -t <ruta/a/carpeta/template> -i <ruta/a/carpeta/de/imagenes> -o <ruta/a/careta/de/salida> [-g <height>] [-w <width>]
```

* __generate_model.py__ Crea y entrena una Convolutional Neural Networks ([LeNet](http://yann.lecun.com/exdb/lenet/)) utilizando el dataset generado por *prepare_dataset.py*. Entrega un modelo entrenado, su gráfico de entrenamiento y varios checkpoints.
    * __-i__: Ruta a la carpeta con el dataset de entrenamiento, la que contiene las carpetas de las clases.
    * __-o__: Ruta donde se guardaran los resultados del script.
    * __-e__: Si se usa esta opción, el entrenamiento se realizará con _early stop_.
    * __-c__: Si se usa esta opción, durante el entrenamiento se guardarán _checkpoints_.
    * __-m__ (opcional): Ruta a un modelo pre-entrenado, este se re-entrenará.
    * __-n__ (opcional): Nombre del modelo. Como default, se genera una id random.
    * __-g__ (opcional): Altura de las imagenes del dataset. Como default se usa 40px.
    * __-w__ (opcional): Ancho de las imagenes del dataset. Como default se usa 40px.

```
$ generate_model.py -i <ruta/a/dataset/de/entrenamieto> -o <ruta/a/carpeta/de/salida> [-e] [-c] [-m <ruta/a/modelo>] [-n <nombre>] [-g <height>] [-w <width>]
```

* __consume_model.py__ Utilizando un template y un modelo ya entrenado, realiza un análisis de sobre una imagen de formulario y entrega un json con los resultados obtenidos.
    * __-t__: Ruta a la carpeta que contiene los datos del template, generada por _prepare_template.py__.
    * __-i__: Ruta a la imagen de un formulario.
    * __-o__: Ruta a la carpeta donde se guardaran los resultados del script.
    * __-m__: Ruta al modelo entrenado.
    * __-f__: Si se usa esta opción, el json mostrará más información de los resultados.
    * __-g__ (opcional): Altura de las imagenes del dataset. Como default se usa 40px.
    * __-w__ (opcional): Ancho de las imagenes del dataset. Como default se usa 40px.

```
$ consume_model.py -t <tuta/a/la/carpeta/del/template> -i <ruta/a/la/imagen> -o <ruta/a/carpeta/de/salida>  -m <ruta/a/modelo> [-f] [-g <height>] [-w <width>]
```


## Workflow

Como usar estos scripts para generar un modelo.
1. Obtener suficientes __imagenes__ de formularios, selecionar una para ser utilizado como __template__.
2. Utilizar __prepare_template.py__.
3. Utilizar __prepare_dataset.py__.
4. Utilizar __generate_model.py__.

Esto nos entrega un modelo el cual puede ser evaluado y comparado con otros modelos generados. Luego de seleccionar al mejor, este se puede utilizar con el script __consume_model.py__.

## Arquitectura de CNN

Para la clasificación se utiliza un modelo de red neuronal convolucional basado en la arquitectura ([LeNet](http://yann.lecun.com/exdb/lenet/))  implementado en [Keras](https://keras.io/)

<img src="https://gitlab.com/sborquez/checkboxes/raw/master/checkbox/LeNet.png" alt="LeNet" height="520">