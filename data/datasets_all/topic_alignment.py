import os.path
import pickle
from datasets import load_from_disk
from bertopic import BERTopic
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

    ds = load_from_disk(ds_path)

    if os.path.exists("all_texts_embeddings.pkl"):
        with open("all_texts_embeddings.pkl", 'rb') as p:
            embeddings = pickle.load(p)
    else:
        embeddings = embedding_model.encode(ds["CleanedText"], show_progress_bar=True)
        # TODO: add metadata for embedding model name?
        with open("all_texts_embeddings.pkl", 'wb') as p:
            pickle.dump(embeddings, p)

    if not os.path.exists(ds_path+"_topics"):
        ds = ds.remove_columns("__index_level_0__")

        iter = ds.iter(batch_size=10000)
        topics = []
        i = 0
        for subset in iter:
            ts, probs = loaded_model.transform(documents=subset['CleanedText'],
                                               embeddings=embeddings[i:i + len(subset["CleanedText"])])
            topics.extend(list(ts))
            i += len(subset["CleanedText"])

        ds = ds.add_column("Topic", topics)
        ds.save_to_disk(ds_path + "_topics")
