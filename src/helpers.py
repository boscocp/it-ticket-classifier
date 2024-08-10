import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


nltk.download("stopwords", quiet=True)
stemmer = nltk.stem.SnowballStemmer("english")
stopwords = nltk.corpus.stopwords.words("english")


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = " ".join([w for w in text.split() if w not in stopwords])
    text = text.encode("ascii", "ignore").decode("utf-8")
    return text


class TFIDFExtrator(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, max_features: int = None) -> None:
        if max_features is None:
            max_features = 128
        self.column = column
        self.tfidf = TfidfVectorizer(max_features=max_features)

    def fit(self, X, y=None):
        self.tfidf.fit(X)
        return self

    def transform(self, X):
        return self.tfidf.transform(X)
