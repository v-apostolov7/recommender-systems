import pandas as pd
from surprise import Dataset, Reader


def load_and_prepare_data(file_path):
    # Четем CSV
    df = pd.read_csv(file_path)
    df = df.sample(n=100000, random_state=42)
    # Дефинираме скалата на оценките
    reader = Reader(rating_scale=(0.5, 5.0))
    # Зареждаме за Surprise (UserId, MovieId, Rating)
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    return data
