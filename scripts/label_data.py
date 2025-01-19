import re
import pandas as pd


def extract_location(text):
    """
    Extracts location-related information from the given text using a regex pattern.
    """
    pattern = r"((?:አድራሻ|ቁጥር \d+ :-)\s*.*?)(?:\n|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    return " ".join(matches).strip() if matches else None


def extract_price(text):
    """
    Extracts price information from the given text using a regex pattern.
    """
    pattern = r"((?:ዋጋ[:፦-]*\s*)?[\d,]+(?:\s[\d,]+)?\s?ብር)"
    matches = re.findall(pattern, text)
    return ", ".join(matches) if matches else None


def extract_product(msg):
    """
    Extracts product information from the given message, ensuring the first line has no "!" or "?"
    and optionally including a second line if it contains only Amharic text.
    """
    # Split message into lines and take the first two
    lines = msg.split("\n")[:2]

    # First line validation: Allow text but NO "!" or "?"
    if re.search(r"[!?]", lines[0]):  # If ! or ? exists, return None
        return None

    first_line = lines[0].strip()

    # Check if a second line exists and contains only Amharic text
    if len(lines) > 1 and re.match(r"^[\u1200-\u137F\s]+$", lines[1]):
        product_text = f"{first_line}\n{lines[1].strip()}"
    else:
        product_text = first_line  # Ignore second line if it doesn't match criteria

    return product_text if product_text else None


def label_data_conll(df, output_file):
    """
    Labels the dataset in CoNLL format for Named Entity Recognition (NER).

    Parameters:
        df (pd.DataFrame): DataFrame containing 'cleaned_message', 'location', 'price', and 'product' columns.
        output_file (str): Path to save the labeled data in CoNLL format.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            if (
                pd.isna(row["cleaned_message"])
                or pd.isna(row["location"])
                or pd.isna(row["price"])
                or pd.isna(row["product"])
            ):
                continue

            words = row["cleaned_message"].split()
            location_words = str(row["location"]).split()
            price_words = str(row["price"]).split()
            product_words = str(row["product"]).split()

            labeled_sentence = []
            for i, word in enumerate(words):
                if word in location_words:
                    label = "B-LOC" if location_words.index(word) == 0 else "I-LOC"
                elif word in price_words:
                    label = "B-PRICE" if price_words.index(word) == 0 else "I-PRICE"
                elif word in product_words:
                    label = (
                        "B-PRODUCT" if product_words.index(word) == 0 else "I-PRODUCT"
                    )
                else:
                    label = "O"

                labeled_sentence.append(f"{word} {label}")

            f.write(
                "\n".join(labeled_sentence) + "\n\n"
            )  # Blank line between sentences


def load_conll_to_dataframe(conll_file):
    """
    Loads CoNLL formatted labeled data into a DataFrame with message_no, token, and label.

    Parameters:
        conll_file (str): Path to the CoNLL file to be loaded.

    Returns:
        pd.DataFrame: DataFrame containing message_no, token, and label.
    """
    message_num = 1
    tokens = []
    labels = []
    message_nums = []

    with open(conll_file, "r", encoding="utf-8") as f:
        sentence = []
        for line in f:
            line = line.strip()
            if line:
                # Split the token and label
                token, label = line.split()
                sentence.append((token, label))
            else:
                # If blank line, end of sentence
                for token, label in sentence:
                    tokens.append(token)
                    labels.append(label)
                    message_nums.append(message_num)
                sentence = []
                message_num += 1

        # In case the last sentence doesn't end with a blank line
        for token, label in sentence:
            tokens.append(token)
            labels.append(label)
            message_nums.append(message_num)

    # Create DataFrame
    df = pd.DataFrame({"message_num": message_nums, "token": tokens, "label": labels})
