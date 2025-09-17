from datasets import load_from_disk


def load_dataset(dataset_path, test=True):
    """
    Load a dataset at given path
    :param test: reduce length of dataset if testing
    :param dataset_path: Path where raw unchunked dataset is
    :return: loaded dataset
    """
    dataset = load_from_disk(dataset_path=dataset_path)
    if test:
        dataset = dataset.filter(lambda example: "dbnl" in example['ID'])
    print(f"Loaded {len(dataset)} entries from {dataset_path}")
    return dataset


def cosine_similarity(a, b):
    """
    Cosine similarity between vectors a and b
    """
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)
