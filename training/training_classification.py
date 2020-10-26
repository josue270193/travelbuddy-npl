#!/usr/bin/env python3
# coding: utf8
"""Entrenador del clasificador si es una review o no utilizando a spaCy
y un set de datos categorizado manualmente de una serie de comentarios extraido de TripAdvisor.

Version: 1.0
Proyecto: TravelBuddy
"""
from __future__ import unicode_literals, print_function

import csv
import os
import random
from pathlib import Path

import plac
import spacy
from spacy.util import minibatch, compounding


@plac.annotations(
    model=("Nombre del Modelo. Por defecto, es un modelo vacio 'es'.", "option", "m", str),
    new_model_name=("Nombre del nuevo modelo.", "option", "nm", str),
    output_dir=("Directorio de salida", "option", "o", Path),
    n_texts=("Cantidad de textos que se tomara para el entrenamiento", "option", "t", int),
    n_iter=("Numero de iteraciones de entrenamiento", "option", "n", int),
    init_tok2vec=("Pesos del preentrenamiento tok2vec", "option", "t2v", Path),
)
def main(model=None, new_model_name="travelbuddy", output_dir="travelbuddy_model", n_iter=100, n_texts=2000,
         init_tok2vec=None):
    """ Ejecuta el entrenador de clasificacion personalizado para los textos de turismo """
    random.seed(0)

    if model is not None:
        nlp = spacy.load(model)  # se carga existente modelo de spaCy
        print("Modelo cargado: '%s'" % model)
    else:
        nlp = spacy.blank("es")  # se crea un modelo vacio del lenguaje
        print("Modelo vacio 'es' creado")

    # Se agrega el proceso del clasificador de texto si no existe
    # nlp.create_pipe se registra con spaCy
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    # En cambio, se obtiene para agregar las etiquetas
    else:
        textcat = nlp.get_pipe("textcat")

    # Se agrega las etiquetas al clasificador de texto
    textcat.add_label("REVIEW")
    textcat.add_label("QUESTION")

    # Se carga los datos de entrenamiento
    print("Se carga los datos de entrenamiento...")
    (train_texts, train_cats), (dev_texts, dev_cats) = load_data()
    train_texts = train_texts[:n_texts]
    train_cats = train_cats[:n_texts]
    print(
        "Using {} examples ({} training, {} evaluation)".format(
            n_texts, len(train_texts), len(dev_texts)
        )
    )
    train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))

    # Obtiene los nombre de los otros procesos para que se desactiven durante el entrenamiento
    pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # Solamente se entrena el proceso "textcat"
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        batch_sizes = compounding(4.0, 32.0, 1.001)
        if init_tok2vec is not None:
            with init_tok2vec.open("rb") as file_:
                textcat.model.tok2vec.from_bytes(file_.read())
        print("Se entrena el modelo...")
        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
        for i in range(n_iter):
            losses = {}
            # Se crea un recopilado de ejemplos para ser usado como un minibatch de spaCy
            random.shuffle(train_data)
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # Se evalua los datos de prueba y se separa de los datos obtenido en load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print(
                "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # Se muestra una tabla simple
                    losses["textcat"],
                    scores["textcat_p"],
                    scores["textcat_r"],
                    scores["textcat_f"],
                )
            )

    # Se prueba el modelo entrenado
    test_text = "Hola, donde recomendarian parar a descansar? Muchas gracias por tu consejo"
    doc = nlp(test_text)
    print(test_text, doc.cats)

    # se guarda el modelo en el directorio de salida
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # renombra el model
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Modelo guardado en: ", output_dir)

        # # test the saved model
        # print("Cargando a partir:", output_dir)
        # nlp2 = spacy.load(output_dir)
        # doc2 = nlp2(test_text)
        # print(test_text, doc2.cats)


def load_data(limit=0, split=0.8):
    """Cargar los datos."""
    data_path = "/../data/out/cats_all.txt"
    with open(os.path.dirname(__file__) + data_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        train_data = list(reader)

    random.shuffle(train_data)
    train_data = train_data[-limit:]
    labels, texts = zip(*train_data)
    labels = [int(i) for i in labels]
    cats = [{"REVIEW": bool(y), "QUESTION": not bool(y)} for y in labels]
    split = int(len(train_data) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "QUESTION":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}


if __name__ == "__main__":
    spacy.require_gpu()
    plac.call(main)
