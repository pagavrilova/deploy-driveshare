import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_data(cars_data):
    """Preprocess the data by filling missing values and encoding categorical data."""
    cars_data["Год до"].fillna(cars_data["Год от"], inplace=True)
    label_encoders = {}
    for column in ["ID_MARK", "Класс", "Страна"]:
        le = LabelEncoder()
        cars_data[column] = le.fit_transform(cars_data[column])
        label_encoders[column] = le
    return cars_data, label_encoders


def calculate_similarity(cars_data):
    """Calculate the cosine similarity matrix for the dataset."""
    features = cars_data[["ID_MARK", "Класс", "Страна"]].values
    year_range = cars_data["Год до"] - cars_data["Год от"]
    year_range_scaled = (year_range - year_range.min()) / (
        year_range.max() - year_range.min()
    )
    features = np.hstack([features, year_range_scaled.values.reshape(-1, 1)])
    return cosine_similarity(features)


def find_similar_cars(model_id, cars_data, similarity_matrix, top_n=5):
    """Find top N similar cars based on the MODEL_ID."""
    if model_id not in cars_data["MODEL_ID"].values:
        return "MODEL_ID not found in the dataset."

    model_index = cars_data[cars_data["MODEL_ID"] == model_id].index[0]
    similar_indices = np.argsort(-similarity_matrix[model_index])[1 : top_n + 1]
    return cars_data.iloc[similar_indices]


# Example usage
if __name__ == "__main__":
    filepath = "./data_gen/Cars Base.csv"
    cars_data = pd.read_csv(filepath)
    cars_data, _ = preprocess_data(cars_data)
    similarity_matrix = calculate_similarity(cars_data)
    model_id = "AUDI_S6"
    similar_cars = find_similar_cars(model_id, cars_data, similarity_matrix)
    print(list(similar_cars[["MODEL_ID"]]["MODEL_ID"]))
