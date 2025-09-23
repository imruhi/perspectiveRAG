def evaluate_results(results_df, cross_model=True, cross_lanugage=True, metrics=None):
    """
    Evaluate responses based on metric
    :param results_df: df where answers are saved
    :param metrics: which metrics to use
    :param cross_model: if we want to evaluate responses amongst models
    :param cross_lanugage: if we want to evaluate responses amongst languages
    :return:
    """

    if metrics is None:
        metrics = ["BERTScore", "TER", "rouge", "cosine"]
