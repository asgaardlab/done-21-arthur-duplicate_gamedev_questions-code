def get_X_y(df):
    """Creates the X (features) and y (target) datasets
    to be used by the classifier
    """
    X = df.set_index(["candidate_id", "dup_id"])
    y = X["is_dup"]
    X = X.drop(columns=["is_dup", "score"])
    return X, y
