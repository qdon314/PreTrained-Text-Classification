from utils.data_set_utils import show_random_elements
from datasets import load_dataset

def test_show_random_elements_runs():
    dataset = load_dataset("imdb")
    show_random_elements(dataset["train"])
    assert True  # If the function runs without errors, the test passes
