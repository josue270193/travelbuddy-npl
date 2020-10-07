#!/usr/bin/env python3
# coding: utf8
"""Genera un archivo json con los datos para ser entrenado a partir de una serie de json de datos
que contiene un mapa de entidades y su posicion en el texto.

Version: 1.0
Proyecto: TravelBuddy
"""
import csv
from os import listdir
from os.path import isfile, join

# Carga los archivos json de la carpeta directorio
from pathlib import Path


def load_file_data(dir_path, data_trained):
    # Se obiene los archivos json del directorio por parametro
    train_files = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith(".txt")]

    pre_data = []
    data = []
    for filename_file in train_files:
        print(filename_file)
        with open(filename_file, encoding='utf8') as file_data:
            train = file_data.readlines()

        for d in train:
            clean_text = d.replace('\t', '').replace('\n', '')
            if clean_text not in data_trained and clean_text not in pre_data:
                print(d)
                is_review = bool(int(input("Es una review?: ") or 0))
                pre_data.append(clean_text)
                data.append(str(int(is_review)) + d)
    return data


def load_file_pretrained(filename_pretrained):
    data_pretrained_aux = []
    with open(filename, 'r', encoding='utf8') as file_pretrained:
        data_file_pretrained = csv.reader(file_pretrained, delimiter='\t')
        for [_, text] in list(data_file_pretrained):
            data_pretrained_aux.append(text)
    return data_pretrained_aux


# Compila todos los archivos de entrenamiento en un solo archivo
if __name__ == "__main__":
    # Reviso que el archivo final exista
    filename = 'cats.txt'
    file_path = Path(filename)
    data_pretrained = []
    if file_path.exists():
        data_pretrained = load_file_pretrained(filename)

    train_data = load_file_data("cats", data_pretrained)

    with open(filename, 'a', encoding='utf8') as files:
        files.writelines(train_data)

    # print(TRAIN_DATA)
    print("Archivo creado", filename)
