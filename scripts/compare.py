import os
import time
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from seqeval.metrics import classification_report
from datasets import load_dataset
from collections import defaultdict


def align_labels_and_predictions(texts, labels, tokenizer, nlp_pipeline):
    """Tokenizes input text, runs model predictions, and aligns them with true labels."""
    y_true, y_pred = [], []

    for text, label_seq in zip(texts, labels):
        # Convert label IDs to actual label names
        true_labels = label_seq

        # Get model predictions
        predictions = nlp_pipeline(text)
        pred_labels = [
            prediction[0]["entity"] if prediction else "O" for prediction in predictions
        ]

        y_true.append(true_labels)
        y_pred.append(pred_labels)

    return [y_true], [y_pred]


def per_entity_report(y_true, y_pred):
    """Generates per-entity F1, Precision, Recall"""
    report = classification_report(y_true, y_pred, output_dict=True)

    entity_scores = {}
    for entity, metrics in report.items():
        if isinstance(metrics, dict):  # Ignore overall metrics
            entity_scores[entity] = {
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-score": metrics["f1-score"],
            }

    return entity_scores


def measure_inference_time(nlp_pipeline, texts):
    start_time = time.time()
    for text in texts:
        _ = nlp_pipeline(text)  # Run inference
    end_time = time.time()
    avg_time = (end_time - start_time) / len(texts)

    return avg_time


def get_model_size(model_name):
    path = f"./{model_name}"
    os.system(
        f"git clone https://huggingface.co/{model_name} {path}"
    )  # Download model if not local
    size = sum(
        os.path.getsize(os.path.join(path, f))
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    )
    return size / (1024 * 1024)  # Convert bytes to MB
