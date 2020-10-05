#!/usr/bin/env python3
# coding: utf8
"""Prueba del modelo custom con el ofrecido por spacy para el NER

Version: 1.0
Proyecto: TravelBuddy
"""
import json
import os
import random
from os import listdir
from os.path import isfile, join

import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer


# Obtiene las estadisticas de rendimiento de un modelo NER de los datos de prueba.
def evaluate(ner_model, examples):
    scorer = Scorer()
    for text, annot in examples:
        doc_gold_text = ner_model.make_doc(text)
        gold = GoldParse(doc_gold_text, entities=annot["entities"])
        pred_value = ner_model(text)
        scorer.score(pred_value, gold)
    return scorer.scores


def load_file_data(dir_path):
    # Se obiene los archivos json del directorio por parametro
    files = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith(".json")]

    data = []
    for filename_file in files:
        # print(filename_file)
        with open(filename_file, encoding='utf8') as file_data:
            train = json.load(file_data)

        for d in train:
            if d['content']:
                data.append((d['content'], {'entities': d['entities']}))

    random.shuffle(data)
    return data


# Obtiene las predicciones que un modelo NER produce de los textos de los datos de prueba.
def get_predictions(ner_model, examples):
    preds = []
    for text, annot in examples:
        doc = ner_model(text)
        entities = [[ent.start_char, ent.end_char, ent.label_] for ent in doc.ents]
        preds.append((text, {"entities": entities}))
    return preds


if __name__ == "__main__":
    # Carga el modelo custom
    model_path = input("Enter your Model Name: ") or "travelbuddy_model"
    model_path = os.path.dirname(__file__) + "/../" + model_path
    custom_nlp = spacy.load(model_path)
    # Carga un modelo base de spaCy
    default_nlp = spacy.load("es_core_news_lg")
    # Carga los datos de pruebas
    test_data_path = "train"
    test_data = load_file_data(test_data_path)
    # Se realiza las pruebas de los modelos
    custom_eval = evaluate(custom_nlp, test_data)
    default_eval = evaluate(default_nlp, test_data)

    print("\n")
    print("CUSTOM STATS")
    print("precision:", custom_eval["ents_p"])
    print("recall:", custom_eval["ents_r"])
    print("f-score:", custom_eval["ents_f"])
    print("per type:", custom_eval["ents_per_type"])
    print("---")
    print("DEFAULT STATS")
    print("precision:", default_eval["ents_p"])
    print("recall:", default_eval["ents_r"])
    print("f-score:", default_eval["ents_f"])
    print("per type:", default_eval["ents_per_type"])
    print("\n")
