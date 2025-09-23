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

    for model_name in models:

        df1 = answers[answers["model"] == model_name].reset_index(drop=True)

        for question in questions:

            df2 = df1[df1["question_mapped"] == question].reset_index(drop=True)

            for i in range(PARAMS["repeat"]):

                dutch_ans = df2[df2["language"] == "NL"].reset_index(drop=True)[f"repeat_{i + 1}"].iloc[0]
                french_ans = df2[df2["language"] == "FR"].reset_index(drop=True)[f"repeat_{i + 1}"].iloc[0]
                english_ans = df2[df2["language"] == "EN"].reset_index(drop=True)[f"repeat_{i + 1}"].iloc[0]
                dutch_ans_b = df2[df2["language"] == "NL"].reset_index(drop=True)[f"baseline_repeat_{i + 1}"].iloc[0]
                french_ans_b = df2[df2["language"] == "FR"].reset_index(drop=True)[f"baseline_repeat_{i + 1}"].iloc[0]
                english_ans_b = df2[df2["language"] == "EN"].reset_index(drop=True)[f"baseline_repeat_{i + 1}"].iloc[0]

                nl_fr = cosine_sim(dutch_ans, french_ans)
                fr_en = cosine_sim(french_ans, english_ans)
                en_nl = cosine_sim(english_ans, dutch_ans)
                nl_fr_b = cosine_sim(dutch_ans_b, french_ans_b)
                fr_en_b = cosine_sim(french_ans_b, english_ans_b)
                en_nl_b = cosine_sim(english_ans_b, dutch_ans_b)

                nl_fr_r = rougeL_fmeasure(dutch_ans, french_ans)
                fr_en_r = rougeL_fmeasure(french_ans, english_ans)
                en_nl_r = rougeL_fmeasure(english_ans, dutch_ans)
                nl_fr_rb = rougeL_fmeasure(dutch_ans_b, french_ans_b)
                fr_en_rb = rougeL_fmeasure(french_ans_b, english_ans_b)
                en_nl_rb = rougeL_fmeasure(english_ans_b, dutch_ans_b)

                all_measures.append([model_name, question, nl_fr, fr_en, en_nl, nl_fr_r, fr_en_r, en_nl_r, False])
                all_measures.append(
                    [model_name, question, nl_fr_b, fr_en_b, en_nl_b, nl_fr_rb, fr_en_rb, en_nl_rb, True])

    all_measures_df = pd.DataFrame(all_measures,
                                   columns=["Model", "Question", "NL_FR_C", "FR_EN_C", "EN_NL_C", "NL_FR_R", "FR_EN_R",
                                            "EN_NL_R", "fBaseline"])
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
