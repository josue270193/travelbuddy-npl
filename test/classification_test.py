#!/usr/bin/env python3
# coding: utf8
"""Prueba del modelo custom con el ofrecido por spacy para el textcat

Version: 1.0
Proyecto: TravelBuddy
"""
import os
import random
from os import listdir
from os.path import isfile, join

import spacy


# Se hace la prueba del modelo con los ejemplos.
def evaluate(textcat_model, examples):
    for test_text in examples:
        doc = textcat_model(test_text)
        print(test_text, doc.cats)


def load_file_data(dir_path):
    # Se obiene los archivos json del directorio por parametro
    files = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith(".txt")]

    data = []
    for filename_file in files:
        # print(filename_file)
        with open(filename_file, encoding='utf8') as file_data:
            for text in file_data.readlines():
                data.append(text)
    random.shuffle(data)
    return data


if __name__ == "__main__":
    # Carga el modelo custom
    model_path = input("Enter your Model Name: ") or "travelbuddy_model"
    spacy_default_models = {"es_core_news_lg", "en_core_web_sm", "en_core_web_lg"}
    if model_path not in spacy_default_models:
        model_path = os.path.dirname(__file__) + "/../training/" + model_path
    custom_nlp = spacy.load(model_path)
    # Carga los datos de pruebas
    test_data_path = "cats"
    test_data = load_file_data(test_data_path)
    # Se realiza las pruebas de los modelos
    evaluate(custom_nlp, test_data)
