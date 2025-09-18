from datasets import Dataset, load_dataset, concatenate_datasets
import string
import os.path
from pandas import concat, DataFrame
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

YEARS = [1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819]

# chunk texts
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-2", chunk_size=300, chunk_overlap=100
)


def preprocess(dataset, path):
    print("Cleaning data since cleaned version not found")
    all_strs = set(" ".join(dataset["Text"]))
    cleaned_text = []

    remove_chars = [x for x in all_strs if
                    not x.isalnum() and
                    x not in string.punctuation and
                    not x.isspace()]

    for text in dataset["Text"]:
        x = text
        for i in remove_chars:
            x = x.replace(i, "")
        x = x.replace('\n-', '')
        x = re.sub(r'\n+', ' ', x)
        cleaned_text.append(x)

    dataset = dataset.add_column("CleanedText", cleaned_text)
    dataset.save_to_disk(path + "-cleaned")
    return dataset


def chunk_examples(examples):
    new_texts = []
    new_metadata = {k: [] for k in examples.keys() if k != "Text"}

    for i in range(len(examples["Text"])):
        text = examples["Text"][i]
        # collect the metadata for this row
        metadata = {k: examples[k][i] for k in examples.keys() if k != "Text"}

        # Split text
        chunks = text_splitter.split_text(text)

        # Add one copy of metadata for each chunk
        for chunk in chunks:
            new_texts.append(chunk)
            for k, v in metadata.items():
                new_metadata[k].append(v)

    return {
        "Text": new_texts,
        **new_metadata
    }


def preprocess_american(dataset):
    datasets_ = []

    for split in dataset:
        dataset[split] = dataset[split].add_column("Year", [int(split)] * len(dataset[split]))
        datasets_.append(dataset[split])

    dataset = concatenate_datasets(datasets_)
    dataset = dataset.rename_columns({"article": "Text"})

    chunked_dataset = dataset.map(
        chunk_examples,
        batched=True,
    )

    return chunked_dataset


DATASET = pd.read_csv("C:/Users/imruh/Documents/perspectiveRAG/data/remove_headings/train_test_remove_heading.csv",
                      converters={'points': pd.eval})


def train_classifier(k=3):
    X = pd.DataFrame(DATASET['points'].to_list())
    y = DATASET['boolRemove'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=clf.classes_)
    disp.plot()
    plt.savefig("C:/Users/imruh/Documents/perspectiveRAG/data/remove_headings/CM_headingremoval.png")
    pickle.dump(clf, open('C:/Users/imruh/Documents/perspectiveRAG/data/remove_headings/KNN_headingremoval', 'wb'))


def load_headingremoval_model(filepath):
    if not os.path.isfile(filepath):
        train_classifier()
    return pickle.load(open(filepath, 'rb'))


def bool_remove(points,
                model_filepath="C:/Users/imruh/Documents/perspectiveRAG/data/remove_headings/KNN_headingremoval"):
    model = load_headingremoval_model(model_filepath)
    X = pd.DataFrame(points.to_list()).fillna(0).iloc[:, :48]
    predictions = model.predict(X)
    return predictions


class Delpher:
    """
    An instance of Delpher data
    """

    def __init__(self):
        self.path = "C:/Users/imruh/Documents/perspectiveRAG/data/datasets_all/delpher-subset"
        if os.path.exists(self.path + "-cleaned"):
            self.dataset = Dataset.load_from_disk(self.path + "-cleaned")
        else:
            dataset = Dataset.load_from_disk(self.path)
            self.dataset = preprocess(dataset, self.path)
        self.path = self.path + "-cleaned"


class DBNL:
    """
    An instance of DBNL data
    """

    def __init__(self):
        self.path = "C:/Users/imruh/Documents/perspectiveRAG/data/datasets_all/dbnl-subset"
        if os.path.exists(self.path + "-cleaned"):
            self.dataset = Dataset.load_from_disk(self.path + "-cleaned")
        else:
            dataset = Dataset.load_from_disk(self.path)
            self.dataset = preprocess(dataset, self.path)
        self.path = self.path + "-cleaned"


class AmericanStories:
    """
    An instance of American stories data
    :param: year_list, range of years (dataset is very big)
    """

    def __init__(self, year_list=None):
        if year_list is None:
            year_list = YEARS
        self.path = "C:/Users/imruh/Documents/perspectiveRAG/data/datasets_all/americanstories-subset"
        if os.path.exists(self.path + "-cleaned"):
            self.dataset = Dataset.load_from_disk(self.path + "-cleaned")
        else:
            path = "dell-research-harvard/AmericanStories"
            dataset = load_dataset(path, "subset_years", year_list=[str(x) for x in year_list], trust_remote_code=True)
            dataset = preprocess_american(dataset)
            self.dataset = preprocess(dataset, self.path)
        self.path = self.path + "-cleaned"


class Wikipedia:
    """
    An instance of web scraped wiki data
    :param: language, specified language
    """

    def __init__(self, language):
        self.path = "C:/Users/imruh/Documents/perspectiveRAG/data/datasets_all/wikipedia-subset"
        if os.path.exists(self.path + "-cleaned"):
            self.dataset = Dataset.load_from_disk(self.path + "-cleaned").filter(lambda example:
                                                                                 example["Language"] == language)
        else:
            dataset = Dataset.load_from_disk(self.path)
            self.dataset = preprocess(dataset, self.path)
        self.path = self.path + "-cleaned"


# TODO: remove legislations which are not in range of correct years
class Plakaatboeken:
    """
    An instance of the Nederlands Indische Plakaatboeken from KB/Annemieke
    """

    def __init__(self):
        self.path = "C:/Users/imruh/Documents/perspectiveRAG/data/datasets_all/plakaatboeken"
        if os.path.exists(self.path + "-cleaned"):
            self.dataset = Dataset.load_from_disk(self.path + "-cleaned")
        else:
            with open("C:/Users/imruh/Documents/perspectiveRAG/data/raw/placaatboek/all_vers_preprocessed", 'rb') as b:
                dfs = pickle.load(b)
            df = concat(dfs, ignore_index=True)
            ds = Dataset.from_pandas(DataFrame({"Year": [int(a) for a in df['year']], "Date": df['date'],
                                                "Text": df['legislation'], "Book": df['book']}))
            self.dataset = preprocess(ds, self.path)

        self.path = self.path + "-cleaned"


class BLBooks:
    """
    An instance of the public domain british library books
    """

    def __init__(self):
        self.path = "C:/Users/imruh/Documents/perspectiveRAG/data/datasets_all/blbooks-subset"

        if not os.path.exists(self.path):
            ds = load_dataset("TheBritishLibrary/blbooks", trust_remote_code=True, split='train')
            filtered_ds = ds.filter(
                lambda example: example['date'].year in YEARS and len(example['text']) > 100 and example[
                    'Language_1'] == 'English')
            filtered_ds.save_to_disk(self.path)

        if os.path.exists(self.path + "-cleaned"):
            self.dataset = Dataset.load_from_disk(self.path + "-cleaned")
            self.dataset = self.dataset.rename_columns({"text": "Text"})
        else:
            dataset = Dataset.load_from_disk(self.path)
            dataset = dataset.rename_columns({"text": "Text"})
            self.dataset = preprocess(dataset, self.path)
            self.dataset = self.dataset.map(
                chunk_examples,
                batched=True,
            )
        self.path = self.path + "-cleaned"
