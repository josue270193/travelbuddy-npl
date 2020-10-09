# travelbuddy-npl
Proyecto de PFI para la carrera de Ingenieria en Informatica para la UADE.

## Instalacion
Se requiere de Python version 3.0+ y un sistema con una GPU Nvidia con sus drivers CUDA activos*
 
*para poder realizar el entranamiento y la evaluacion mas rapido.

Revisar el archivo `requirements.txt` 

## Crear release
`python -m spacy package -f travelbuddy_model dist/`

`cd dist/es_travelbuddy-x.x.x/`

`python setup.py sdist`

`pip install dist/es_travelbuddy-x.x.x.tar.gz`
