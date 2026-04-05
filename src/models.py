from surprise import SVD, KNNBasic, NormalPredictor


def get_models():
    return {
        "Random": NormalPredictor(),
        "KNN": KNNBasic(),
        "SVD": SVD()
    }
