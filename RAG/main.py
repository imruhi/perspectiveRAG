import pickle
from advanced_rag import AdvancedRAG
import pandas as pd
from multiprocessing import freeze_support
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from os import path
from datasets import Dataset, Features, Value, Sequence, load_from_disk, concatenate_datasets
from questions import load_questions
import json
import time
import glob
from evaluate_gen import (get_amonst_lang_eval, get_baseline_lang_eval, get_reranked_doc_eval, compare_within,
                          get_amonst_lang_setting_eval, turn_text_to_conll)


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

        # one KB and one vec storage for each language (also topics filtered out in both)
        aRag = AdvancedRAG(embedding_model_name=params["embedding_model_name"], max_new_tokens=params["max_new_tokens"],
                           reader_model_name=model_name, cross_encoder_name=params["cross_encoder_name"],
                           dataset_path=params["dataset_path"], temperature=params["temperature"],
                           topics=params["topics"], bwiki=params["bwiki"])

        for language in params["languages"]:

            all_answers = []
            reranked_doc_ids = []
            retrieved_doc_ids = []
            model_names = []
            all_questions = []
            all_languages = []
            baseline_answers = []

            # save per language
            results_path = (f"t{params['temperature']}_new{params['max_new_tokens']}_"
                            f"{params['embedding_model_name'].split('/')[-1]}_"
                            f"{params['cross_encoder_name'].split('/')[-1]}_{model_name.split('/')[-1]}_{language}")

            print(f"On {model_name} for {language}", flush=True)
            vec_save = params["vec_save"] + "_" + language + ".pkl"

            questions = load_questions(language)
            languages = [language] * len(questions)
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
            aRag.prompt_model_baseline(repeat=params["repeat"])
            aRag.prompt_model(brerank=True if params["rerank"] == "True" else False, top_k=params["top_k"],
                              rerank_k=params["rerank_k"], repeat=params["repeat"])

            for query in aRag.questions:
                all_answers.append(query.answers)
                all_questions.append(query.question)
                all_languages.append(query.language)
                baseline_answers.append(query.baseline_answers)
                model_names.append(model_name.split('/')[-1])
                retrieved_doc_ids.append(query.get_retrieved_doc_ids())
                if params["rerank"] == "True":
                    reranked_doc_ids.append(query.get_reranked_doc_ids())

            aRag.questions = []

            # save answers based on language
            df = pd.DataFrame({"question": all_questions,
                               "answers": all_answers,
                               "baseline_answer": baseline_answers,
                               "reranked_doc_ids": reranked_doc_ids,
                               "retrieved_doc_ids": retrieved_doc_ids,
                               "language": all_languages,
                               "model": model_names})

            features = Features({
                "question": Value("string"),
                "answers": Sequence(Value("string")),
                "baseline_answer": Sequence(Value("string")),
                "reranked_doc_ids": Sequence(Value("string")),
                "retrieved_doc_ids": Sequence(Value("string")),
                "language": Value("string"),
                "model": Value("string"),
            })

            ds = Dataset.from_pandas(df, features=features)
            ds.save_to_disk(results_path)

    end = time.time()
    print(f"{(end - start) / 60} mins", flush=True)


def get_answers_df():
    paths = glob.glob("evaluation/t0*")
    print(paths)
    answers = [load_from_disk(x) for x in paths]
    answers = concatenate_datasets(answers)

    answers = answers.to_pandas()

    questions_NL = answers[answers["language"] == 'NL']["question"].unique()
    questions_EN = answers[answers["language"] == 'EN']["question"].unique()
    questions_FR = answers[answers["language"] == 'FR']["question"].unique()

    question_map_NL = {questions_NL[f]: f"Question {f + 1}" for f in range(len(questions_NL))}
    question_map_EN = {questions_EN[f]: f"Question {f + 1}" for f in range(len(questions_EN))}
    question_map_FR = {questions_FR[f]: f"Question {f + 1}" for f in range(len(questions_FR))}

    question_map = dict(question_map_NL, **question_map_EN, **question_map_FR)
    print(question_map)
    answers["question_mapped"] = [question_map[x] for x in answers["question"]]

    with open("evaluation/question_map.pkl", 'wb') as f:
        pickle.dump(question_map, f)

    return answers


def evaluate():
    answers = get_answers_df()
    get_amonst_lang_eval(answers).to_csv("evaluation/among_lang_metrics.csv")
    get_baseline_lang_eval(answers).to_csv("evaluation/between_lang_baseline_metrics.csv")
    get_reranked_doc_eval(answers).to_csv("evaluation/retrieved_docs_metrics.csv")
    compare_within(answers).to_csv("evaluation/within_baseline_non_baseline.csv")
    get_amonst_lang_setting_eval(answers).to_csv("evaluation/all_comparisons.csv")


def answers_ud():
    answers = get_answers_df()
    turn_text_to_conll(answers=answers)


if __name__ == "__main__":
    # freeze_support()  # for synchronity issues in using FAISS to make vec store
    # main()            # generate answers
    evaluate()
