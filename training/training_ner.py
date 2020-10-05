#!/usr/bin/env python3
# coding: utf8
"""Entrenador NER para los comentarios de Tripadvisor utilizando spaCy
se toma la base del ejemplo provisto por spaCy en su pagina.

Para mas detalle revisar:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Version: 1.0
Proyecto: TravelBuddy
"""
from __future__ import unicode_literals, print_function

import json
import os
import random
import warnings
from pathlib import Path

import plac
import spacy
from spacy.util import minibatch, compounding


@plac.annotations(
    model=("Nombre del Modelo. Por defecto, es un modelo vacio 'es'.", "option", "m", str),
    new_model_name=("Nombre del nuevo modelo.", "option", "nm", str),
    output_dir=("Directorio de salida", "option", "o", Path),
    n_iter=("Numero de iteraciones de entrenamiento", "option", "n", int),
)
def main(model=None, new_model_name="travelbuddy", output_dir="travelbuddy_model", n_iter=100):
    """ Ejecuta el entrenador de anotaciones NER personalizado para los textos de turismo """
    random.seed(0)

    if model is not None:
        nlp = spacy.load(model)  # se carga existente modelo de spaCy
        print("Modelo cargado: '%s'" % model)
    else:
        nlp = spacy.blank("es")  # se crea un modelo vacio del lenguaje
        print("Modelo vacio 'es' creado")

    # se crea los componente necesario y se agregan al flujo
    # nlp.create_pipe funciona para los componentes que son registrados en spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # sino, lo obtenemos y podemos agregar las etiquetas
    else:
        ner = nlp.get_pipe("ner")

    # Se carga los datos de entrenamiento
    print("Se carga los datos de entrenamiento...")
    train_data = load_data()

    # se agrega las etiquetas
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # se obtiene los nombres de los otros flujos para desactivarlos durante el entrenamiento
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # se entrena solamente el "NER"
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # se filtra las advertencias por el alineamiento erroneo en las entity
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # se resetea y se inicializa el peso de forma aleatoria - pero solamente si estamos entrenando un modelo nuevo
        optimizer = nlp.begin_training()
        batch_sizes = compounding(4.0, 32.0, 1.001)
        for i in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            # se arma un minibatch de spaCy de ejemplo a partir de los datos de entrenamiento
            batches = minibatch(train_data, batch_sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # los textos
                    annotations,  # las anotaciones
                    sgd=optimizer,
                    drop=0.2,  # abandono: dificulta la memorización de datos
                    losses=losses,
                )
            print("Losses:", losses)

    # # se prueba el modelo entrenado
    # for text, _ in train_data:
    #     doc = nlp(text)
    #     print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])

    # se guarda el modelo en el directorio de salida
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # renombra el model
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Modelo guardado en: ", output_dir)

        # # se prueba el modelo guardado
        # print("Cargando a partir:", output_dir)
        # nlp2 = spacy.load(output_dir)
        # for text, _ in TRAIN_DATA:
        #     doc = nlp2(text)
        #     print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])


def load_data():
    """Cargar los datos."""
    data_path = "/../data/train.json"
    with open(os.path.dirname(__file__) + data_path, 'r') as f:
        train_data = json.load(f)

    return train_data


if __name__ == "__main__":
    spacy.require_gpu()
    plac.call(main)
