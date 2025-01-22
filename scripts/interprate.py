import shap
import torch
import numpy as np


class NERWrapper:
    """
    A wrapper class for Named Entity Recognition (NER) models to be used with SHAP for interpretability.

    Args:
        model (torch.nn.Module): The pre-trained NER model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer corresponding to the model.

    Methods:
        predict(texts): Tokenizes the input texts, performs inference, and returns aggregated probabilities.
    """

    def __init__(self, model, tokenizer):
        """
        Initializes the NERWrapper with the model and tokenizer.

        Args:
            model (torch.nn.Module): The NER model.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for text preprocessing.
        """
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, texts):
        """
        Tokenizes the input texts, performs inference, and returns aggregated probabilities.

        Args:
            texts (str or list of str): The input texts for prediction.

        Returns:
            np.ndarray: Aggregated probabilities for each text, averaged over the token dimension.
        """
        # SHAP may provide inputs as a NumPy array of dtype=object, so convert them properly
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()  # Convert to list
        if isinstance(texts, list):
            texts = [str(text) for text in texts]  # Ensure elements are strings
        elif isinstance(texts, str):
            texts = [texts]

        # Tokenize input
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs).logits

        # Aggregate the model output for all tokens (e.g., average over token predictions)
        probs = torch.nn.functional.softmax(outputs, dim=-1)
        aggregated_probs = probs.mean(
            dim=1
        )  # Average over the sequence length dimension

        return aggregated_probs.numpy()


def predict_proba(texts, model, tokenizer):
    """
    Computes the probability distribution over entity classes for each token in the input texts.

    Args:
        texts (str or list of str): The input text or a list of texts for prediction.
        model (torch.nn.Module): The pre-trained NER model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer corresponding to the model.

    Returns:
        np.ndarray: A NumPy array of shape (batch_size, seq_len * num_classes) containing
                    the probability distribution for each token in each input text.
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).logits  # Shape: (batch_size, seq_len, num_classes)

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(outputs, dim=-1)
    return probs.numpy().reshape(len(texts), -1)
