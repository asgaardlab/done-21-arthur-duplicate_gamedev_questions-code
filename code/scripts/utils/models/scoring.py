def recall_rate_k_scorer(df, col, k):
    """Scores a dataframe based on the column using the recall-rate@k measure"""

    def recall_rate_k(df):
        return df.sort_values(col, ascending=False)[:k]["is_dup"].any()

    return df.groupby("dup_id").apply(recall_rate_k).mean()


def predict_probabilities(estimator, X, y):
    """Uses the classifier to predict the classification probabilities using the set of features X"""
    df = X.reset_index()[["dup_id", "candidate_id"]]
    df = df.set_index(["candidate_id", "dup_id"])

    df["pred"] = [p[1] for p in estimator.predict_proba(X)]
    df["is_dup"] = y
    df = df.reset_index()
    
    return df


def multiple_k_scorer(estimator, X, y):
    """Uses the estimator to predict values of X, and evaluates it using multiple
    recall-rates measures
    """
    df = predict_probabilities(estimator, X, y)

    return {
        "rr@5": recall_rate_k_scorer(df, "pred", 5),
        "rr@10": recall_rate_k_scorer(df, "pred", 10),
        "rr@20": recall_rate_k_scorer(df, "pred", 20),
    }
