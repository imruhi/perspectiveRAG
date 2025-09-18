import pickle

from advanced_rag import AdvancedRAG
import pandas as pd
from multiprocessing import freeze_support
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from os import path
from datasets import Dataset

# TODO: set param py file
vec_save = "data/faiss_index_test.pkl"

# rag params (experiment params)
model_name = "Qwen/Qwen2-7B-Instruct"
cross_encoder_name = "colbert-ir/colbertv2.0"  # "cross-encoder/ms-marco-MiniLM-L6-v2"
embedding_model_name = "all-MiniLM-L6-v2"
max_new_tokens = 300
temperature = 0.3
dataset_path = "data/RAG_DB"
topics = [89, 92, 135]
# generation params
rerank = True
top_k = 20
rerank_k = 3

# results save name
# temp_max-new-tokens_embedding-model_cross-encoder_langmodel
results_path = (f"results/t{temperature}_new{max_new_tokens}_{embedding_model_name.split('/')[-1]}_"
                f"{cross_encoder_name.split('/')[-1]}_{model_name.split('/')[-1]}")

# query params
# TODO: hardcoded but change to editable txt file?
questions = [
    "Wie was Napoleon Bonaparte?",
    "Was Napoleon Bonaparte een held?",
    "Was Napoleon Bonaparte een tiran?",
    "Was Napoleon Bonaparte een schurk?",
    "Hoe werd Napoleon Bonaparte keizer van Frankrijk?",
    "Werd Napoleon's aanspraak op keizerschap in Europa gesteund?",
    "Welke invloed had Napoleon's heerschappij op Engeland en Nederland?",
    "Was de impact van Napoleons heerschappij op Engeland en Nederland destructief?",
]

languages = ['nl'] * len(questions)


def main():
    aRag = AdvancedRAG(embedding_model_name=embedding_model_name, max_new_tokens=max_new_tokens,
                       reader_model_name=model_name, cross_encoder_name=cross_encoder_name,
                       dataset_path=dataset_path, temperature=temperature, topics=topics)

    # vector database init based on cosine

    if path.exists(vec_save):
        print("Loading VD")
        with open(vec_save, 'rb') as f:
            aRag.set_vector_store(pickle.load(f))
    else:
        # in main due to synchronity issues
        print("Making VD")
        aRag.vector_base = FAISS.from_documents(
            aRag.knowledge_base, aRag.embedding_model, distance_strategy=DistanceStrategy.COSINE
        )
        with open(vec_save, 'wb') as f:
            pickle.dump(aRag.vector_base, f)

    aRag.set_questions(questions, languages)
    aRag.prompt_model(brerank=rerank, top_k=top_k, rerank_k=rerank_k)

    answers = []
    reranked_doc_ids = []
    retrieved_doc_ids = []
    for query in aRag.questions:
        print(f"Query: {query.question}")
        print(f"Answer: {query.answer}")
        print()
        answers.append(query.answer.replace('\n', ' '))
        retrieved_doc_ids.append(query.get_retrieved_doc_ids())
        if rerank:
            reranked_doc_ids.append(query.get_reranked_doc_ids())

    # TODO: experiment with multiple answers?
    ds = Dataset.from_pandas(pd.DataFrame({"question": questions, "answer": answers,
                                           "reranked_doc_ids": reranked_doc_ids,
                                           'retrieved_doc_ids': retrieved_doc_ids,
                                           "language": languages}))
    ds.save_to_disk(results_path)


if __name__ == "__main__":
    freeze_support()  # for synchronity issues in using FAISS to make vec store
    main()
