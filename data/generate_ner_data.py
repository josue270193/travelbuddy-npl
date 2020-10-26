#!/usr/bin/env python3
# coding: utf8
"""Genera un archivo json con los datos para ser entrenado a partir de una serie de json de datos
que contiene un mapa de entidades y su posicion en el texto.

Version: 1.0
Proyecto: TravelBuddy
"""
import json
from os import listdir
from os.path import isfile, join


# Carga los archivos json de la carpeta directorio
def load_file_data(dir_path):
    # Se obiene los archivos json del directorio por parametro
    files = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith(".json")]

    data = []
    for filename_file in files:
        print(filename_file)
        with open(filename_file, encoding='utf8') as file:
            train = json.load(file)

        for d in train:
            if d['content']:
                data.append((d['content'], {'entities': d['entities']}))
    return data


# Compila todos los archivos de entrenamiento en un solo archivo
if __name__ == "__main__":
    train_data = load_file_data("ner/files")

    filename = 'out/ner_all.json'
    with open(filename, 'w', encoding='utf8') as file_json:
        json.dump(train_data, file_json, ensure_ascii=False)

    # print(TRAIN_DATA)
    print("Archivo creado", filename)
