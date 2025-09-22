import os.path
import pickle
import pandas as pd

from datasets import load_from_disk
from bertopic import BERTopic
from collections import Counter
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer


def align_text(embed_model_name="sentence-transformers/distiluse-base-multilingual-cased-v1",
               ds_path="all-texts-metadata",
               topic_model_path="../../Topic_alignment/topic_model/reduced_model_14"):
    """
    Add the topic col in dataset
    :param embed_model_name: model used for embedding text
    :param ds_path: path where hf dataset is
    :param topic_model_path: path to saved topic model
    :return: none (save dataset with topic column)
    """
    embedding_model = SentenceTransformer(embed_model_name,
                                          model_kwargs={"torch_dtype": "float16"})
    loaded_model = BERTopic.load(topic_model_path, embedding_model=embedding_model)

    if os.path.exists(ds_path+'_embeds'):
        ds = load_from_disk(ds_path+'_embeds')
    else:
        ds = load_from_disk(ds_path)
        embeddings = embedding_model.encode(ds["CleanedText"], show_progress_bar=True)
        # TODO: add metadata for embedding model name?
        ds = ds.add_column("Embeddings", embeddings)
        ds.save_to_disk(ds_path+"_embeds")

    topics, probs = loaded_model.transform(documents=ds['CleanedText'], embeddings=ds["Embeddings"])
    ds = ds.add_column("Topic", topics)
    ds.save_to_disk(ds_path+"_embeds_topics")

