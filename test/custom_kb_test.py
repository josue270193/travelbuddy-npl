#!/usr/bin/env python3
# coding: utf8
"""Prueba del modelo custom con el ofrecido por spacy para el textcat

Version: 1.0
Proyecto: TravelBuddy
"""
import os

import spacy


# Se hace la prueba del modelo con los ejemplos.
def evaluate(ner_model, test_text):
    doc = ner_model(test_text)
    entities = [(e.text, e.label_, e.kb_id_) for e in doc.ents]
    print(entities)


if __name__ == "__main__":
    # Carga el modelo custom
    model_path = input("Enter your Model Name: ") or "travelbuddy_model"
    spacy_default_models = {"es_core_news_lg", "en_core_web_sm", "en_core_web_lg"}
    if model_path not in spacy_default_models:
        model_path = os.path.dirname(__file__) + "/../" + model_path
    custom_nlp = spacy.load(model_path)
    # Se ingresa el texto custom
    test_data = input("Text to test: ")
    # Se realiza las pruebas de los modelos
    evaluate(custom_nlp, test_data)
