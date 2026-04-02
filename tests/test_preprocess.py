import importlib
import sys
from unittest.mock import MagicMock, patch


def load_preprocess_module():
    sys.modules.pop("src.data_pipeline.preprocess", None)
    with patch("nltk.download", return_value=True):
        return importlib.import_module("src.data_pipeline.preprocess")


def test_clean_text_normalizes_twitter_content():
    preprocess = load_preprocess_module()

    with patch.object(preprocess.stopwords, "words", return_value=["is", "a"]):
        processor = preprocess.TextPreprocessor()

    text = "Hello @alice! Visit https://example.com #Amazing 123 &amp; WOW"

    cleaned = processor.clean_text(text)

    assert cleaned == "hello visit amazing wow"

def test_preprocess_batch_applies_each_step_in_order():
    preprocess = load_preprocess_module()

    with patch.object(preprocess.stopwords, "words", return_value=["the"]), patch.object(
        preprocess.nltk, "word_tokenize", side_effect=lambda text: text.split()
    ), patch.object(preprocess, "WordNetLemmatizer") as lemmatizer_cls:
        lemmatizer = MagicMock()
        lemmatizer.lemmatize.side_effect = lambda token: token.rstrip("s")
        lemmatizer_cls.return_value = lemmatizer

        processor = preprocess.TextPreprocessor()
        processed = processor.preprocess_batch(
            ["The cats are RUNNING!!!", "@bob likes dogs #pets"]
        )

    assert processed == ["cat are running", "like dog pet"]
