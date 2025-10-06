from sentence_transformers import SentenceTransformer, util
import json
import pandas as pd
from rouge_score import rouge_scorer
import itertools
from datasets import load_from_disk
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.parse.util import taggedsents_to_conll
import os

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

with open("../params.json", 'r') as f:
    PARAMS = json.load(f)

SMODEL = SentenceTransformer(PARAMS["embedding_model_name"])


def cosine_sim(sent1, sent2):
    emb1 = SMODEL.encode(sent1, convert_to_tensor=True)
    emb2 = SMODEL.encode(sent2, convert_to_tensor=True)
    return round(float(util.pytorch_cos_sim(emb1, emb2).cpu().detach().numpy()[0][0]), 4)


def rougeL_fmeasure(sent1, sent2):
    return round(scorer.score(sent1, sent2)["rougeL"].fmeasure, 4)


# TODO probably a faster way to do it since they all use the same for loops but a bit confusing to code

def get_amonst_lang_eval(answers, lang_pairs=None):
    """
    scores amongst languages and baseline
    :param lang_pairs: pairs of language for evaluation
    :param answers: answers df
    :return: measures df
    """

    # Better way to do it since I'm less tired
    # one to many comparison

    all_measures = []
    models = list(answers["model"].unique())
    questions = list(answers["question_mapped"].unique())

    if lang_pairs is None:
        # use all language pairs
        lang_pairs = list(itertools.combinations(answers["language"].unique(), 2))
    print(lang_pairs)
    for model_name in models:

        df1 = answers[answers["model"] == model_name].reset_index(drop=True)

        for question in questions:

            df2 = df1[df1["question_mapped"] == question].reset_index(drop=True)

            for pair in lang_pairs:
                all_answers_1 = list(df2[df2["language"] == pair[0]].reset_index(drop=True)["answers"].iloc[0])
                baseline_answers_1 = list(
                    df2[df2["language"] == pair[0]].reset_index(drop=True)["baseline_answer"].iloc[0])
                all_answers_2 = list(df2[df2["language"] == pair[1]].reset_index(drop=True)["answers"].iloc[0])
                baseline_answers_2 = list(
                    df2[df2["language"] == pair[1]].reset_index(drop=True)["baseline_answer"].iloc[0])

                pair_answers = list(itertools.product(all_answers_1, all_answers_2))
                pair_answers_baseline = list(itertools.product(baseline_answers_1, baseline_answers_2))

                # product pairs with context
                for x in pair_answers:
                    cosine = cosine_sim(x[0], x[1])
                    all_measures.append([model_name, question, cosine, False, f"{pair[0]}-{pair[1]}"])
                # product pairs with baselines
                for x in pair_answers_baseline:
                    cosineb = cosine_sim(x[0], x[1])
                    all_measures.append([model_name, question, cosineb, True, f"{pair[0]}-{pair[1]}"])

    all_measures_df = pd.DataFrame(all_measures,
                                   columns=["Model", "Question", "Cosine", "fBaseline", "Language1-Language2"])
    return all_measures_df


def get_amonst_lang_setting_eval(answers, lang_pairs=None):
    """
    scores amongst all combinations constant for model
    :param lang_pairs: pairs of language for evaluation
    :param answers: answers df
    :return: measures df
    """

    # Better way to do it since I'm less tired
    # one to many comparison

    all_measures = []
    models = list(answers["model"].unique())
    questions = list(answers["question_mapped"].unique())

    if lang_pairs is None:
        # use all language pairs
        lang_pairs = list(itertools.combinations(answers["language"].unique(), 2))
    print(lang_pairs)
    for model_name in models:

        df1 = answers[answers["model"] == model_name].reset_index(drop=True)

        for question in questions:

            df2 = df1[df1["question_mapped"] == question].reset_index(drop=True)

            for pair in lang_pairs:
                all_answers_1 = list(df2[df2["language"] == pair[0]].reset_index(drop=True)["answers"].iloc[0])
                baseline_answers_1 = list(
                    df2[df2["language"] == pair[0]].reset_index(drop=True)["baseline_answer"].iloc[0])
                all_answers_2 = list(df2[df2["language"] == pair[1]].reset_index(drop=True)["answers"].iloc[0])
                baseline_answers_2 = list(
                    df2[df2["language"] == pair[1]].reset_index(drop=True)["baseline_answer"].iloc[0])

                pair_answers = list(itertools.product(all_answers_1, baseline_answers_2))
                pair_answers2 = list(itertools.product(all_answers_2, baseline_answers_1))

                # product pairs with context
                for x in pair_answers:
                    cosine = cosine_sim(x[0], x[1])
                    all_measures.append([model_name, question, cosine, f"{pair[0]}-{pair[1]}"])
                # product pairs with baselines
                for x in pair_answers2:
                    cosineb = cosine_sim(x[0], x[1])
                    all_measures.append([model_name, question, cosineb, f"{pair[0]}-{pair[1]}"])

    all_measures_df = pd.DataFrame(all_measures,
                                   columns=["Model", "Question", "Cosine", "Language1-Language2"])
    return all_measures_df


def get_baseline_lang_eval(answers, languages=None):
    """
    bertscore between languages and also between language and baseline
    :param languages: languages for evaluation
    :param answers: answers df
    :return: measures df
    """

    # Better way to do it
    # one to many comparison

    all_measures = []
    models = list(answers["model"].unique())
    questions = list(answers["question_mapped"].unique())

    if languages is None:
        languages = list(answers["language"].unique())

    for model_name in models:

        df1 = answers[answers["model"] == model_name].reset_index(drop=True)

        for question in questions:

            df2 = df1[df1["question_mapped"] == question].reset_index(drop=True)

            for lang in languages:
                all_answers = list(df2[df2["language"] == lang].reset_index(drop=True)["answers"].iloc[0])
                baseline_answers = list(df2[df2["language"] == lang].reset_index(drop=True)["baseline_answer"].iloc[0])

                pair_answers = list(itertools.product(all_answers, baseline_answers))
                for x in pair_answers:
                    cosine = cosine_sim(x[0], x[1])
                    rouge = rougeL_fmeasure(x[0], x[1])
                    all_measures.append([model_name, question, lang, cosine, rouge])

    all_measures_df = pd.DataFrame(all_measures,
                                   columns=["Model", "Question", "Language", "Cosine", "Rouge"])

    return all_measures_df


def compare_within(answers, columns=None, languages=None):
    if columns is None:
        columns = ["answers", "baseline_answer"]

    all_measures = []
    models = list(answers["model"].unique())
    questions = list(answers["question_mapped"].unique())

    if languages is None:
        languages = list(answers["language"].unique())

    for model_name in models:

        df1 = answers[answers["model"] == model_name].reset_index(drop=True)

        for question in questions:

            df2 = df1[df1["question_mapped"] == question].reset_index(drop=True)

            for lang in languages:
                for column in columns:
                    all_answers = list(df2[df2["language"] == lang].reset_index(drop=True)[column].iloc[0])
                    pairs = list(itertools.combinations(all_answers, 2))
                    for p in pairs:
                        cosine = cosine_sim(p[0], p[1])
                        rouge = rougeL_fmeasure(p[0], p[1])
                        all_measures.append([model_name, question, lang, cosine, rouge, column])

    all_measures_df = pd.DataFrame(all_measures,
                                   columns=["Model", "Question", "Language", "Cosine", "Rouge", "Answer"])
    return all_measures_df


def get_reranked_doc_eval(answers, dspath="all-texts-metadata_topics", languages=None):
    """
    Evaluate the top k docs used in context for prompt
    :param answers: answers df
    :param dspath: path to original texts
    :param languages: languages to evaluate
    :return: df with rouge and bertscore measures
    """
    texts_ds = load_from_disk(dspath)
    texts_dict = {id_: text for id_, text in zip(texts_ds["ID"], texts_ds["CleanedText"])}

    if languages is None:
        languages = list(answers["language"].unique())

    models = list(answers["model"].unique())
    questions = list(answers["question_mapped"].unique())
    all_measures = []
    for model_name in models:

        df1 = answers[answers["model"] == model_name].reset_index(drop=True)

        for question in questions:

            df2 = df1[df1["question_mapped"] == question].reset_index(drop=True)

            for lang in languages:
                all_answers = list(df2[df2["language"] == lang].reset_index(drop=True)["reranked_doc_ids"].iloc[0])
                pairs = list(itertools.combinations(all_answers, 2))
                for p in pairs:
                    cosine = cosine_sim(texts_dict[p[0]], texts_dict[p[1]])
                    rouge = rougeL_fmeasure(texts_dict[p[0]], texts_dict[p[1]])
                    all_measures.append([model_name, question, lang, cosine, rouge])

    all_measures_df = pd.DataFrame(all_measures,
                                   columns=["Model", "Question", "Language", "Cosine", "Rouge"])
    return all_measures_df


def turn_text_to_conll(answers, ):
    """
    For UD profilling, turn each answer into one document of lines in conll-u format
    :param answers: answers df
    :return: return a file which can be saved
    """

    if not os.path.exists("augmented_answers"):
        os.makedirs("augmented_answers")
    if not os.path.exists("baseline_answers"):
        os.makedirs("baseline_answers")

    for _, row in answers.iterrows():
        idx = row["idx"]
        i = 1
        question = row["question_mapped"].replace(" ", "")
        for a, b in zip(row["answers"], row["baseline_answer"]):
            sentences = [pos_tag(word_tokenize(sent)) for sent in sent_tokenize(a)]
            sentencesb = [pos_tag(word_tokenize(sent)) for sent in sent_tokenize(a)]

            tagged_sents = taggedsents_to_conll(sentences)
            tagged_sentsb = taggedsents_to_conll(sentencesb)

            text = "".join([x for x in tagged_sents])
            textb = "".join([x for x in tagged_sentsb])

            f_name = f"augmented_answers/{idx}_{question}_{i}.txt"
            with open(f_name, "w", encoding="utf-8") as f:
                f.write(text)
            f_name = f"baseline_answers/{idx}_{question}_{i}.txt"
            with open(f_name, "w", encoding="utf-8") as f:
                f.write(textb)

            i += 1

