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
import time


# TODO: probably a way to adjust params so it goes faster but can't be bothered
def main():
    # load parameters
    with open("params.json", 'r') as f:
        params = json.load(f)

    print(params, flush=True)
    print('\n', flush=True)

    # results save name: temp_max-new-tokens_embedding-model_cross-encoder
    start = time.time()

    for model_name in params["model_names"]:

        all_answers = []
        reranked_doc_ids = []
        retrieved_doc_ids = []
        model_names = []
        all_questions = []
        all_languages = []
        results_path = (f"t{params['temperature']}_new{params['max_new_tokens']}_"
                        f"{params['embedding_model_name'].split('/')[-1]}_"
                        f"{params['cross_encoder_name'].split('/')[-1]}_{model_name.split('/')[-1]}")

        # one KB and one vec storage for each language (also topics filtered out in both)
        aRag = AdvancedRAG(embedding_model_name=params["embedding_model_name"], max_new_tokens=params["max_new_tokens"],
                           reader_model_name=model_name, cross_encoder_name=params["cross_encoder_name"],
                           dataset_path=params["dataset_path"], temperature=params["temperature"],
                           topics=params["topics"])

        for language in params["languages"]:
            print(f"On {model_name} for {language}", flush=True)
            vec_save = params["vec_save"] + "_" + language + ".pkl"

            questions = load_questions(language)
            languages = [language] * len(questions)
            ############# TEST #############
            questions = questions[:1]
            languages = languages[:1]
            ################################
            print(len(questions), flush=True)

            aRag.set_language(language)
            aRag.prepare()  # change KB to curr language one, so model isn't loaded every turn
            # vector database init based on cosine
            # pickle save/load, can't find anything better
            if path.exists(vec_save):
                print("Loading VD", flush=True)
                with open(vec_save, 'rb') as f:
                    aRag.set_vector_store(pickle.load(f))
            else:
                # in main due to synchronity issues
                print("Making VD", flush=True)
                aRag.vector_base = FAISS.from_documents(
                    aRag.knowledge_base, aRag.embedding_model, distance_strategy=DistanceStrategy.COSINE
                )
                with open(vec_save, 'wb') as f:
                    pickle.dump(aRag.vector_base, f)

            aRag.set_questions(questions, languages)
            aRag.prompt_model(brerank=True if params["rerank"] == "True" else False, top_k=params["top_k"],
                              rerank_k=params["rerank_k"])

            for query in aRag.questions:
                print(f"Query: {query.question}", flush=True)
                print(f"Answer: {query.answer}", flush=True)
                print()
                all_answers.append(query.answer.replace('\n', ' '))
                all_questions.append(query.question)
                all_languages.append(query.language)
                model_names.append(model_name.split('/')[-1])
                retrieved_doc_ids.append(query.get_retrieved_doc_ids())
                if params["rerank"] == "True":
                    reranked_doc_ids.append(query.get_reranked_doc_ids())
            aRag.questions = []

        # save answers based on language
        # TODO: experiment with multiple answers?
        df = pd.DataFrame({"question": all_questions,
                           "answer": all_answers,
                           "reranked_doc_ids": reranked_doc_ids,
                           'retrieved_doc_ids': retrieved_doc_ids,
                           "language": all_languages,
                           "model": model_names})

        ds = Dataset.from_pandas(df)
        ds.save_to_disk(results_path)

    end = time.time()
    print(f"{(end - start) / 60} mins", flush=True)


if __name__ == "__main__":
    freeze_support()  # for synchronity issues in using FAISS to make vec store
    main()
