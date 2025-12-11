from datasets import load_dataset
from deepchecks.nlp import TextData
import pandas as pd

def test_textdata_from_hf_dataset():
    ds = load_dataset("ag_news", split="train[:20]")

    td = TextData.from_huggingface(ds)

    assert isinstance(td, TextData)
    assert len(td) == 20
    assert isinstance(td.text, (list, pd.Series, pd.Index))
