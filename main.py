#!/usr/bin/env python3
# coding: utf8
"""Modelo principal donde se expone un clasificador sentimental

Version: 1.0
Proyecto: TravelBuddy
"""

import es_travelbuddy
import plac
import spacy
from classifier import *


class Result:
    text = ""
    entities = []
    score_sentimental = 0.0
    score_review = 0.0
    score_question = 0.0


def evaluate_sentimental_text(text_sentimental):
    result = clf_model.predict(text_sentimental)
    return result


def evaluate_ner_text(doc):
    entities = [{'type': ent.label_, 'text': ent.text} for ent in doc.ents]
    return entities


def evaluate_cats_text(doc):
    cats = doc.cats
    return cats


def create_result(text_to_evaluate, result_sentimental, result_ner, result_cats):
    new_result = Result()
    new_result.text = text_to_evaluate or ""
    new_result.score_sentimental = result_sentimental or 0.0
    new_result.score_review = result_cats['REVIEW'] or 0.0
    new_result.score_question = result_cats['QUESTION'] or 0.0
    new_result.entities = result_ner or []
    return new_result


@plac.annotations(
    text_to_evaluate=("Texto a evaluar.", "option", "t", str),
)
def main(text_to_evaluate=""):
    doc = nlp_model(text_to_evaluate)
    result_sentimental = evaluate_sentimental_text(text_to_evaluate)
    result_ner = evaluate_ner_text(doc)
    result_cats = evaluate_cats_text(doc)
    result = create_result(text_to_evaluate, result_sentimental, result_ner, result_cats)
    result_json = json.dumps(result.__dict__)
    return print(result_json)


if __name__ == '__main__':
    spacy.require_gpu()
    nlp_model = es_travelbuddy.load()
    clf_model = SentimentClassifier()
    plac.call(main)
