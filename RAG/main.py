import pickle

from advanced_rag import AdvancedRAG
import pandas as pd
from multiprocessing import freeze_support
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from os import path
from datasets import Dataset
from questions import load_questions
import json


# TODO: probably a way to adjust params so it goes faster but can't be bothered
def main():
    # load parameters
    with open("params.json", 'r') as f:
        params = json.load(f)

    # results save name: temp_max-new-tokens_embedding-model_cross-encoder
    results_path = (f"results/t{params['temperature']}_new{params['max_new_tokens']}_"
                    f"{params['embedding_model_name'].split('/')[-1]}_"
                    f"{params['cross_encoder_name'].split('/')[-1]}")

    answers = []
    reranked_doc_ids = []
    retrieved_doc_ids = []
    model_names = []
    questions = load_questions(params['language'])
    languages = [params["language"]] * len(questions)
    vec_save = params["vec_save"] + "_" + params["language"] + ".pkl"

    for model_name in params["model_names"]:
        print(f"On {model_name} for {params['language']}")

        # one KB for all languages but seperate vec storage for languages
        aRag = AdvancedRAG(embedding_model_name=params["embedding_model_name"], max_new_tokens=params["max_new_tokens"],
                           reader_model_name=params["model_name"], cross_encoder_name=params["cross_encoder_name"],
                           dataset_path=params["dataset_path"], temperature=params["temperature"],
                           topics=params["topics"], language=params["language"])

        # vector database init based on cosine
        # pickle save/load, can't find anything better
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
        aRag.prompt_model(brerank=True if params["rerank"] == "True" else False, top_k=params["top_k"],
                          rerank_k=params["rerank_k"])

        for query in aRag.questions:
            print(f"Query: {query.question}")
            print(f"Answer: {query.answer}")
            print()
            answers.append(query.answer.replace('\n', ' '))
            retrieved_doc_ids.append(query.get_retrieved_doc_ids())
            if params["rerank"] == "True":
                reranked_doc_ids.append(query.get_reranked_doc_ids())
        model_names.extend([model_name.split('/')[-1]] * len(questions))

    # save answers based on language
    # TODO: experiment with multiple answers?
    ds = Dataset.from_pandas(pd.DataFrame({"question": questions, "answer": answers,
                                           "reranked_doc_ids": reranked_doc_ids,
                                           'retrieved_doc_ids': retrieved_doc_ids,
                                           "language": languages,
                                           "model": model_names}))
    ds.save_to_disk(results_path)


if __name__ == "__main__":
    freeze_support()  # for synchronity issues in using FAISS to make vec store
    main()
