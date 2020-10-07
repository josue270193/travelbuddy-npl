#!/usr/bin/env python3
# coding: utf8
"""Prueba del modelo custom con el ofrecido por spacy para el textcat

Version: 1.0
Proyecto: TravelBuddy
"""
import os
import re

import spacy
from spacy.gold import biluo_tags_from_offsets
from spacy.tokenizer import Tokenizer


def my_tokenizer(nlp_aux, infix_re_aux):
    return Tokenizer(nlp_aux.vocab,
                     {},
                     infix_finditer=infix_re_aux.finditer
                     )


# Se hace la prueba del modelo con los ejemplos.
def evaluate(ner_model, test_text):
    doc = ner_model(test_text)
    entities = [
        [
            14,
            28,
            "ATTRACTION"
        ],
        [
            115,
            127,
            "ATTRACTION"
        ],
        [
            204,
            219,
            "ACTIVITY"
        ],
        [
            96,
            111,
            "ACTIVITY"
        ],
        [
            305,
            319,
            "ATTRACTION"
        ],
        [
            390,
            402,
            "ATTRACTION"
        ]
    ]
    tags = biluo_tags_from_offsets(doc, entities)
    print(tags)


if __name__ == "__main__":
    # Carga el modelo custom
    model_path = input("Enter your Model Name: ") or "travelbuddy_model"
    if model_path != "es-core-news-lg":
        model_path = os.path.dirname(__file__) + "/../" + model_path
    custom_nlp = spacy.load(model_path)

    # Tokenizador
    infix_re = re.compile(r'''[a-zA-Z]''')
    custom_nlp.tokenizer = my_tokenizer(custom_nlp, infix_re)

    # Se ingresa el texto custom
    test_data = input("Text to test: ")
    # Se realiza las pruebas de los modelos
    evaluate(custom_nlp, test_data)
