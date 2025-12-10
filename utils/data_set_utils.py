import datasets
import random
import pandas as pd
from IPython.display import display, HTML

"""
Utility functions for inspecting and visualizing HuggingFace Datasets.

These helpers are primarily designed for use inside Jupyter notebooks, where
HTML rendering is supported. Functions here allow quick random inspection of
dataset samples, along with automatic decoding of ClassLabel features.
"""

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))