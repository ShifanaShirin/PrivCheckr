import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download once (first time only)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def sentence_segmentation(text: str):
    return sent_tokenize(text)


def tokenize_sentence(sentence: str):
    return word_tokenize(sentence)


def preprocess_text(text: str):
    normalized = normalize_text(text)
    sentences = sentence_segmentation(normalized)

    processed = []
    for sentence in sentences:
        tokens = tokenize_sentence(sentence)
        processed.append({
            "sentence": sentence,
            "tokens": tokens
        })

    return processed
