from sentence_transformers import SentenceTransformer, util
import json
import pickle
import pandas as pd
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

with open("params.json", 'r') as f:
    PARAMS = json.load(f)

SMODEL = SentenceTransformer(PARAMS["embedding_model_name"])


def cosine_sim(sent1, sent2):
    emb1 = SMODEL.encode(sent1, convert_to_tensor=True)
    emb2 = SMODEL.encode(sent2, convert_to_tensor=True)
    return round(float(util.pytorch_cos_sim(emb1, emb2).cpu().detach().numpy()[0][0]), 4)


def rougeL_fmeasure(sent1, sent2):
    return round(scorer.score(sent1, sent2)["rougeL"].fmeasure, 4)


def get_evaluate_measures(answers):
    """
    get semantic similarity and rouge scores
    :param answers: answers df
    :return: measures df
    """

    # There is probably a better way to do this but I'm tired

    all_measures = []
    models = list(answers["model"].unique())
    questions = list(answers["question_mapped"].unique())

    lang_pairs = [["NL", "FR"], ["FR", "EN"], ["EN", "NL"]]

    for model_name in models:

        df1 = answers[answers["model"] == model_name].reset_index(drop=True)

        for question in questions:

            df2 = df1[df1["question_mapped"] == question].reset_index(drop=True)

            for i in range(PARAMS["repeat"]):

                for pair in lang_pairs:

                    ans1 = df2[df2["language"] == pair[0]].reset_index(drop=True)[f"repeat_{i + 1}"].iloc[0]
                    ans2 = df2[df2["language"] == pair[1]].reset_index(drop=True)[f"repeat_{i + 1}"].iloc[0]
                    ans1b = df2[df2["language"] == pair[0]].reset_index(drop=True)[f"baseline_repeat_{i + 1}"].iloc[0]
                    ans2b = df2[df2["language"] == pair[1]].reset_index(drop=True)[f"baseline_repeat_{i + 1}"].iloc[0]

                    cosine = cosine_sim(ans1, ans2)
                    cosineb = cosine_sim(ans1b, ans2b)

                    rouge = rougeL_fmeasure(ans1, ans2)
                    rougeb = rougeL_fmeasure(ans1b, ans2b)

                    all_measures.append([model_name, question, cosine, rouge, False, f"{pair[0]}-{pair[1]}"])
                    all_measures.append([model_name, question, cosineb, rougeb, True, f"{pair[0]}-{pair[1]}"])

    all_measures_df = pd.DataFrame(all_measures,
                                   columns=["Model", "Question", "Cosine", "Rouge", "fBaseline", "Language1-Language2"])
    return all_measures_df


def get_compare_measures(answers):
    """
    Compare language specific answers to baselines
    :return: measures df
    """

    # There is probably a better way to do this but I'm tired

    all_measures = []
    models = list(answers["model"].unique())
    questions = list(answers["question_mapped"].unique())

    for model_name in models:

        df1 = answers[answers["model"] == model_name].reset_index(drop=True)

        for question in questions:

            df2 = df1[df1["question_mapped"] == question].reset_index(drop=True)

            for lang in PARAMS["languages"]:

                for i in range(PARAMS["repeat"]):

                    baseline = df2[df2["language"] == lang].reset_index(drop=True)[f"baseline_repeat_{i + 1}"].iloc[0]
                    context = df2[df2["language"] == lang].reset_index(drop=True)[f"repeat_{i + 1}"].iloc[0]

                    cosine = cosine_sim(baseline, context)
                    rouge = rougeL_fmeasure(baseline, context)
                    all_measures.append(
                        [model_name, question, lang, cosine, rouge])

    all_measures_df = pd.DataFrame(all_measures,
                                   columns=["Model", "Question", "Language", "Cosine", "Rouge"])

    return all_measures_df
