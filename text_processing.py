# core/text_processing.py

import re
from typing import List, Union
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy


nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os

    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text: Union[str, None]) -> str:
        """Basic NLP preprocessing: lowercase, clean, tokenize, remove stopwords, lemmatize"""
        if not isinstance(text, str):
            return ""

        # Lowercase and remove special characters
        text = re.sub(r"[^\w\s]", " ", text.lower())

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words]

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return " ".join(tokens)

    def extract_entities(self, text: str) -> List[dict]:
        """Named Entity Recognition using spaCy"""
        if not isinstance(text, str) or text.strip() == "":
            return []

        doc = nlp(text)
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]
        return entities


if __name__ == "__main__":
    processor = TextProcessor()
    sample = (
        "The steering wheel was replaced at the Ford repair center on 12th Jan 2022."
    )

    print("\nPreprocessed Text:")
    print(processor.preprocess_text(sample))

    print("\nExtracted Entities:")
    for entity in processor.extract_entities(sample):
        print(f"{entity['text']} ({entity['label']})")
