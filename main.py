#!/usr/bin/env python3
# coding: utf8
"""Modelo principal donde se expone un clasificador sentimental

Version: 1.0
Proyecto: TravelBuddy
"""
import getopt

import spacy
from classifier import *

text = ''


class ResultEntity:
    text = ''
    label = ''


class Result:
    score = 0
    entities = []


def calculate_score(textTest):
    result = clf.predict(textTest)
    # print(' ==> %.5f' % result)
    return result


def calcule_text(doc, text):
    entities = []
    for ent in doc.ents:
        resultEntity = ResultEntity()
        resultEntity.text = ent.text
        resultEntity.label = ent.label_
        entities.append(resultEntity)
        # print(ent.text, ent.start_char, ent.end_char, ent.label_)
    score = calculate_score(text)
    result = Result()
    result.score = score
    result.entities = entities
    # print(result)
    return score


def obtain_ner(doc):
    labels = []
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
        labels.append(ent.label_)
    return labels


def check_params(argv):
    global text
    try:
        opts, args = getopt.getopt(argv, "ht:", ["text="])
    except getopt.GetoptError:
        print('main.py -t <texto>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -t <texto>')
            sys.exit()
        elif opt in ("-t", "--text"):
            text = arg
    # print('Texto a Evaluar: ', text)


def main(argv):
    check_params(argv)
    doc = nlp(text)
    textCalculated = calcule_text(doc, text)
    # result = json.dumps(textCalculated.__dict__)
    result = textCalculated

    doc = nlp("Constanza, al igual que Gigiargentina, te recomiendo pasar una noche al menos en La Cumbrecita, es un lugar hermoso que vale mucho la pena conocerlo, yo fuí en auto y ya sabía que debía dejarlo en el estacionamiento y caminar por lo tanto no me incomodé por ello.")
    obtain_ner(doc)

    return print(result)


# calcule_text('El servicio de Trenes es muy bueno ultimamente y tambien informemos que hay promociones en Micro por solo 399 buenos aires mar del plata')
# calcule_text('Porque buenos aires tiene mucho que ofrecer, museos, parques, etc pero a la noche hay espectáculos teatrales, tanguerias, lugares para cenar y bailar, casino.')
# calcule_text('Transporte: para moverte dentro de la ciudad de día, lo mas simple es el Subte para los lugares a los que llega, y ademas hay una amplia red de colectivos')
# calcule_text('Hay otro bus 8 que funciona los domingos, no es semirrápido y dice x Liniers. Tarda más en llegar porque no toma autopista y los deja cerca de Plaza de Mayo también.')
# calcule_text('el tema es que el domingo no funciona la línea 8 no queda más que tomar Uber entonces')

if __name__ == '__main__':
    spacy.prefer_gpu()
    nlp = spacy.load("es_core_news_lg")
    clf = SentimentClassifier()
    main(sys.argv[1:])
