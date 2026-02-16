from datasets import load_from_disk, concatenate_datasets, Dataset
import glob
import re
import os
import pickle
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.parse.util import taggedsents_to_conll
import pandas as pd
import time
from deep_translator import GoogleTranslator
from sentence_splitter import SentenceSplitter


def load_dataset(dataset_path, topics=None, language='NL', bwiki=False):
    """
    Load a dataset at given path, filter for topic and language?
    :param bwiki: true if include wiki docs in returned dataset
    :param language: language subset of dataset
    :param dataset_path: path where raw unchunked dataset is
    :param topics: list of topics (int_numbers) to subset from
    :return: loaded dataset
    """
    dataset = load_from_disk(dataset_path=dataset_path)
    print(f"    Filtering out language")
    if not bwiki:
        dataset = dataset.filter(lambda example: example['SourceLang'] == language and example["Source"] != "wiki")
    else:
        dataset = dataset.filter(lambda example: example['SourceLang'] == language)

    if topics is not None:
        print(f"    Filtering out topics")
        dataset = dataset.filter(lambda example: example['Topic'] in topics)
    print(f"    Loaded {len(dataset)} entries from {dataset_path}")
    return dataset


def get_answers_df():
    paths = glob.glob("evaluation/t0*")
    print(paths)

    def remove_special_chars(sent):
        sent = sent.replace('’', "'")
        x = ''.join(
            ch for ch in sent if ch.isalnum() or ch == ' ' or ch in ["!", '"', "'", ";", ",", ".", "?", ":", "-"])
        return re.sub(' +', ' ', x)

    if not os.path.exists("evaluation/all_answers"):
        answers = [load_from_disk(x) for x in paths]
        answers = concatenate_datasets(answers)

        answers = answers.to_pandas()
        map_ = []
        for language in ["NL", "EN", "FR"]:
            subset = answers[answers["language"] == language]
            questions = subset["question"].unique()
            # TOD0: remove incomplete sentences from answer?
            for f in range(len(questions)):
                map_.append([questions[f], f"Question {f + 1}"])

        question_map = {x[0]: x[1] for x in map_}
        print(question_map)
        answers["question_mapped"] = [question_map[x] for x in answers["question"]]
        # list of lists, remove special chars
        answers["answers"] = [[remove_special_chars(x) for x in y] for y in answers["answers"]]
        answers["baseline_answer"] = [[remove_special_chars(x) for x in y] for y in answers["baseline_answer"]]

        with open("evaluation/question_map.pkl", 'wb') as f:
            pickle.dump(question_map, f)

        Dataset.from_pandas(answers).save_to_disk("evaluation/all_answers")
        return answers
    else:
        return load_from_disk("evaluation/all_answers").to_pandas()


def turn_text_to_conll(answers, ):
    """
    For UD profilling, turn each answer into one document of lines in conll-u format
    :param answers: answers df
    :return: none, file is saved
    """

    if not os.path.exists("augmented_answers"):
        os.makedirs("augmented_answers")
    if not os.path.exists("baseline_answers"):
        os.makedirs("baseline_answers")

    answers["idx"] = [i for i in range(1, len(answers) + 1)]

    for _, row in answers.iterrows():
        idx = row["idx"]
        i = 1
        question = row["question_mapped"].replace(" ", "")
        for a, b in zip(row["answers"], row["baseline_answer"]):
            a = a.replace("*", "").replace("#", "")
            b = b.replace("*", "").replace("#", "")
            sentences = [pos_tag(word_tokenize(sent)) for sent in sent_tokenize(a)]
            sentencesb = [pos_tag(word_tokenize(sent)) for sent in sent_tokenize(b)]

            tagged_sents = taggedsents_to_conll(sentences)
            tagged_sentsb = taggedsents_to_conll(sentencesb)

            text = "".join([x for x in tagged_sents])
            textb = "".join([x for x in tagged_sentsb])

            f_name = f"augmented_answers/{idx}_{question}_{i}.txt"
            with open(f_name, "w", encoding="utf-8") as f1:
                f1.write(text)

            f_name = f"baseline_answers/{idx}_{question}_{i}.txt"
            with open(f_name, "w", encoding="utf-8") as fb:
                fb.write(textb)

            i += 1


def turn_text_to_conll_BL():
    """
    For UD profilling, turn each answer into one document of lines in conll-u format
    :param answers: answers df
    :return: none, file is saved
    """

    if not os.path.exists("BL"):
        os.makedirs("BL")

    dataset = load_from_disk("evaluation/all-texts-metadata_topics")
    dataset = dataset.filter(lambda example: example["Source"] == "blbooks" and example["Topic"] not in [89, 92, 135])

    for language in ['EN', 'FR', 'NL']:
        subset = dataset.filter(lambda example: example['SourceLang'] == language)
        texts = subset.to_pandas()
        texts = texts.sample(n=2000)

        for _, row in texts.iterrows():
            idx = row["ID"]
            a = row["CleanedText"]
            a = a.replace("*", "").replace("#", "")
            sentences = [pos_tag(word_tokenize(sent)) for sent in sent_tokenize(a)]

            tagged_sents = taggedsents_to_conll(sentences)

            text = "".join([x for x in tagged_sents])

            f_name = f"BL/{idx}_{language}.txt"
            with open(f_name, "w", encoding="utf-8") as f1:
                f1.write(text)


def prepare_ud():
    answers = get_answers_df()
    turn_text_to_conll(answers=answers)


def prepare_frame():
    answers = get_answers_df()
    languages = ['EN', 'FR', 'NL']  # answers.language.unique()
    models = answers.model.unique()
    translation_results = pd.DataFrame()
    start = time.time()

    for language in languages:
        splitter = SentenceSplitter(language=language.lower())
        if language != "EN":
            translator = GoogleTranslator(source=language.lower(), target="en")
        for model in models:
            subset = answers[(answers.language == language) & (answers.model == model)].reset_index(drop=True)
            print(f"On {model} for {language}", flush=True)

            # iterate over questions
            for _, row in subset.iterrows():
                print(f"    On Q{_ + 1}", flush=True)
                all_models = []
                conditions = []
                original_languages = []
                translation_sentences = []
                sentences = []
                questions = []
                rag_llms = row["answers"]
                baseline_llms = row["baseline_answer"]
                repeat = 1

                # iterate over repeats
                for rag, baseline in zip(rag_llms, baseline_llms):
                    rag_sentences = splitter.split(rag)
                    baseline_sentences = splitter.split(baseline)

                    try:
                        translated_rag_sentences = translator.translate_batch(rag_sentences) if language != "EN" \
                            else rag_sentences
                        translated_baseline_sentences = translator.translate_batch(
                            baseline_sentences) if language != "EN" \
                            else baseline_sentences
                    except:
                        print("No translation found!")
                        translated_rag_sentences = ["none"] * len(rag_sentences)
                        translated_baseline_sentences = ["none"] * len(baseline_sentences)

                    for x, y in zip([translated_rag_sentences, translated_baseline_sentences], ['rag', 'baseline']):
                        translation_sentences.extend(x)
                        all_models.extend([model] * len(x))
                        original_languages.extend([language] * len(x))
                        temp = [y] * len(x)
                        # condition_#repeeat_#sent
                        conditions.extend([f"{r}_{repeat}_{j}" for j, r in enumerate(temp)])
                        sentences.extend(rag_sentences if y == 'rag' else baseline_sentences)
                        questions.extend([row['question_mapped']] * len(x))
                    repeat += 1

                answer_result = pd.DataFrame({"question": questions,
                                              "translation": translation_sentences, "model": all_models,
                                              "condition": conditions, "language": original_languages,
                                              "sentence": sentences})

                translation_results = pd.concat([answer_result, translation_results])
                # print(len(translation_results), flush=True)
                # print()
                translation_results.to_csv("evaluation/frame_df.csv")

    print(f"{(time.time() - start) / 60} mins", flush=True)


def prepare_frame_source():
    answers = get_answers_df()
    docs = load_from_disk("evaluation/all-texts-metadata_topics").to_pandas()
    texts_dict = {id_: text for id_, text in zip(docs["ID"], docs["CleanedText"])}

    languages = ['EN', 'FR', 'NL']
    models = answers.model.unique()
    translation_results = pd.DataFrame()
    start = time.time()

    for language in languages:
        splitter = SentenceSplitter(language=language.lower())
        if language != "EN":
            translator = GoogleTranslator(source=language.lower(), target="en")
        for model in models:
            subset = answers[(answers.language == language) & (answers.model == model)].reset_index(drop=True)
            print(f"On {model} for {language}", flush=True)

            # iterate over questions
            for _, row in subset.iterrows():
                print(f"    On Q{_ + 1}", flush=True)
                all_models = []
                conditions = []
                original_languages = []
                translation_sentences = []
                sentences = []
                questions = []
                doc_ids = row["reranked_doc_ids"]
                # iterate over repeats
                for rank, id_ in enumerate(doc_ids):
                    chunk = texts_dict[id_]
                    sents = splitter.split(chunk)
                    try:
                        translated_sentences = translator.translate_batch(sents) if language != "EN" \
                            else sents
                    except:
                        print("No translation found!")
                        translated_sentences = ["none"] * len(sents)
                    for y, x in enumerate(translated_sentences):
                        translation_sentences.append(x)
                        all_models.append(model)
                        original_languages.append(language)
                        # rank_#sent
                        conditions.append(f"{rank + 1}_{y}")
                        questions.append(row['question_mapped'])
                    sentences.extend(sents)
                answer_result = pd.DataFrame({"question": questions,
                                              "translation": translation_sentences, "model": all_models,
                                              "condition": conditions, "language": original_languages,
                                              "sentence": sentences})

                translation_results = pd.concat([answer_result, translation_results])
                translation_results.to_csv("evaluation/frame_df_reranked.csv")
            break  # no model level distinction
    print(f"{(time.time() - start) / 60} mins", flush=True)
