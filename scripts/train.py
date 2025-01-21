import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from peft import LoraConfig, get_peft_model
from seqeval.metrics import classification_report


def initialize_model(model_name, num_labels):
    """
    Initializes a token classification model with LoRA (Low-Rank Adaptation) for fine-tuning.

    Args:
        model_name (str): The name of the pre-trained model to use (e.g., 'xlm-roberta-base').
        num_labels (int): The number of labels for the token classification task.

    Returns:
        tuple: A tuple containing:
            - model (PeftModel): The LoRA-adapted model for token classification.
            - tokenizer (AutoTokenizer): The tokenizer corresponding to the pre-trained model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Specify target modules for LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="TOKEN_CLS",
        target_modules=[
            "q_lin",
            "k_lin",
            "v_lin",
            "out_lin",
        ],
    )
    model = get_peft_model(base_model, lora_config)
    return model, tokenizer


def load_conll_file(file_path, seed):
    """
    Loads a dataset from a CoNLL-formatted file and converts it into the Hugging Face Dataset format.

    Args:
        file_path (str): Path to the CoNLL-formatted file.
        seed (int): Random seed for reproducibility when splitting the dataset.

    Returns:
        DatasetDict: A dictionary containing the dataset split into:
            - "train": Training set (80% of the data).
            - "validation": Validation set (10% of the data).
            - "test": Test set (10% of the data).
    """
    sentences = []
    current_sentence = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                token, label = line.split()
                current_sentence.append((token, label))
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []

    # Convert to Hugging Face dataset format
    dataset = Dataset.from_dict(
        {
            "tokens": [[token for token, label in sent] for sent in sentences],
            "ner_tags": [[label for token, label in sent] for sent in sentences],
        }
    )

    # Split the dataset into train, validation, and test sets.
    dataset = dataset.train_test_split(test_size=0.2, seed=seed)
    dataset["validation"] = dataset["test"].train_test_split(test_size=0.5, seed=seed)[
        "train"
    ]
    dataset["test"] = dataset["test"].train_test_split(test_size=0.5, seed=seed)["test"]

    return dataset


def compute_metrics(eval_pred):
    """
    Computes evaluation metrics for a token classification model.

    Args:
        eval_pred (EvalPrediction): An EvalPrediction object containing predictions and labels.

    Returns:
        dict: A dictionary with the classification report and the f1 score.

    This function extracts valid labels (ignoring -100), converts predicted indices to labels,
    and generates a classification report with precision, recall, and F1 scores.
    """
    label_list = [
        "O",
        "B-Product",
        "I-Product",
        "B-LOC",
        "I-LOC",
        "B-PRICE",
        "I-PRICE",
    ]  # Define label_list within the function

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    true_predictions = []
    true_labels = []
    for pred, label in zip(predictions, labels):
        pred_labels = []
        label_labels = []
        for p, l in zip(pred, label):
            if l != -100:
                if p >= len(label_list) or l >= len(label_list):
                    print(f"Invalid Label: pred={p}, true={l}")
                else:
                    pred_labels.append(label_list[p])
                    label_labels.append(label_list[l])
        true_predictions.append(pred_labels)
        true_labels.append(label_labels)

    # Generate classification report
    report = classification_report(true_labels, true_predictions, output_dict=True)

    # Extract and return the overall f1 score
    return {
        "f1": report["weighted avg"][
            "f1-score"
        ],  # Return the f1 score for "metric_for_best_model"
        "classification_report": report,  # Return the full report for other purposes
    }


def tokenize_and_align_labels(examples, tokenizer, model):  # Pass model as argument
    """
    Tokenizes input text and aligns NER labels with tokenized subwords.

    Args:
        examples (dict): A dictionary containing:
            - "tokens" (List[List[str]]): A list of tokenized sentences.
            - "ner_tags" (List[List[str]]): Corresponding NER labels for each token.
        tokenizer: The tokenizer object.
        model: The model object. # Added argument description

    Returns:
        dict: A dictionary containing:
            - Tokenized input with padding and truncation applied.
            - Labels aligned with tokenized subwords, using -100 for subword tokens
              to ignore them during loss computation.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, padding=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(
                    model.config.label2id[label[word_idx]]
                )  # Now uses the model passed as argument
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def train_model(
    model, tokenizer, dataset, output_dir, logging_dir, batch_size=16, epochs=3, lr=5e-5
):
    """
    Trains a token classification model using the Hugging Face Trainer API.

    Args:
        model (PreTrainedModel): The model to train.
        tokenizer (PreTrainedTokenizer): The corresponding tokenizer.
        dataset (DatasetDict): The dataset containing "train" and "validation" splits.
        output_dir (str): Directory to save the trained model.
        logging_dir (str): Directory for training logs.
        batch_size (int, optional): Batch size (default: 16).
        epochs (int, optional): Number of epochs (default: 3).
        lr (float, optional): Learning rate (default: 5e-5).

    This function tokenizes the dataset, trains the model, and saves the best-performing model.
    """
    # tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    # In your train_model function, modify the dataset.map call
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, model),
        batched=True,
    )  # Pass model to tokenize_and_align_labels

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=10,
        report_to=["tensorboard"],
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        push_to_hub=False,
        save_total_limit=2,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def evaluate_model(model, tokenizer, dataset):
    """
    Evaluates a trained token classification model on the test dataset.

    Args:
        model (PreTrainedModel): The trained model to evaluate.
        tokenizer (PreTrainedTokenizer): The corresponding tokenizer.
        dataset (DatasetDict): The dataset containing a "test" split.

    Returns:
        dict: Evaluation metrics, including loss and performance scores.

    This function tokenizes the test dataset, initializes a Trainer, and evaluates the model.
    """
    # Pass model to tokenize_and_align_labels
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, model),
        batched=True,
    )
    trainer = Trainer(model=model)
    results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    return results
