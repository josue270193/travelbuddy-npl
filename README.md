# travelbuddy-npl
Proyecto de PFI para la carrera de Ingenieria en Informatica para la UADE.

## Instalacion
Se requiere de Python version 3.0+ y un sistema con una GPU Nvidia con sus drivers CUDA 10.1 activos*
 
*para poder realizar el entranamiento y la evaluacion mas rapido.

Instalar los componentes dentro del archivo `requirements.txt`, se usa `pip` y para instalarlo usar los pasos de 
https://pip.pypa.io/en/stable/installing/

Posterior a ello se debe abrir una terminal para instalar las siguientes dependencias:
 
- `pip install -U spacy` 
    - `pip install -U spacy[cuda]` en el caso de usar GPU nvidia
- `pip install spanish_sentiment_analysis`
- `python -m spacy download es_core_news_lg`

## Generador de Data de Entrenamiento

Para realizar el entrenamiento se requiere de que los datos de entrenamiento esten en un formato definido, es por ello
que se cuenta con 2 script para compactar y armar los datos en dicho formato.

### Data Entrenamiento NER
Para los datos de entrenamiento para el Reconocimiento de entidades nombradas.
Se debe guardar los archivos `.json` generado por el aplicativo desarrollado en el proyecto `travelbuddy-server` 
dentro de la carpeta `data/ner`.

Posteriormente se procede a ejecutar el archivo dentro de la carpeta `cats` ejecutar el main del archivo
`generate_ner_data.py`

### Data Entrenamiento CATS
Para los datos de entrenamiento para el Clasificador de Texto.
Se debe guardar los archivos `.txt` generado por el aplicativo cuando se extraen los datos del proyecto 
`travelbuddy-server`. Se debe asegurar que tenga los textos en cada linea precedido por una tabulacion.

Posteriormente se procede a ejecutar el archivo dentro de la carpeta `cats` ejecutar el main del archivo
`generate_classification_data.py`

## Entrenamiento

Dentro de la carpeta `data`  

## Crear release
`python -m spacy package -f training/travelbuddy_model dist/`

`cd dist/es_travelbuddy-x.x.x/`

`python setup.py sdist`

`pip install dist/es_travelbuddy-x.x.x.tar.gz`
