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
def load_data(data_path):
    # Se obiene los archivos json en 'data_path'
    files = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f)) and f.endswith(".json")]

    data = []
    for filename_file in files:
        print(filename_file)
        with open(filename_file, encoding='utf8') as file:
            train = json.load(file)

        for d in train:
            data.append((d['content'], {'entities': d['entities']}))
    return data


# Compila todos los archivos de entrenamiento en un solo archivo
if __name__ == "__main__":
    train_data = load_data("files")

    filename = 'train.json'
    with open(filename, 'w', encoding='utf8') as file_json:
        json.dump(train_data, file_json, ensure_ascii=False)

    # print(TRAIN_DATA)
    print("Dumped to", filename)
