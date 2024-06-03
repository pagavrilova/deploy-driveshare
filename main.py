import sqlite3
import pandas as pd
from flask import Flask, request
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from models.cars_similarity import (
    preprocess_data,
    calculate_similarity,
    find_similar_cars,
)
from models.colaborative_filtration import recommend_colaborative_models

app = Flask(__name__)


def get_db_connection():
    conn = sqlite3.connect("./db/rentdb.db")
    conn.row_factory = sqlite3.Row
    return conn


# Инициализация колаборативной фильтрации
conn = get_db_connection()
user_data = pd.read_sql_query("SELECT * FROM users_activities", conn)
cars_data = pd.read_sql_query("SELECT * FROM cars", conn)
conn.close()

# Кодирование идентификаторов пользователей и моделей для использования в матрице
user_encoder = LabelEncoder()
model_encoder = LabelEncoder()

user_data["user_id_encoded"] = user_encoder.fit_transform(user_data["user_id"])
user_data["model_id_encoded"] = model_encoder.fit_transform(user_data["model_id"])

# Создание разреженной матрицы взаимодействий
interaction_matrix = csr_matrix(
    (
        user_data["result_of_activity"],
        (user_data["user_id_encoded"], user_data["model_id_encoded"]),
    ),
    shape=(len(user_encoder.classes_), len(model_encoder.classes_)),
)

# Рассчитываем косинусное сходство между моделями
model_similarity = cosine_similarity(
    interaction_matrix.T
)  # Транспонируем для сравнения моделей


# Инициализации content based рекомендаций
# Создание разреженной матрицы взаимодействий
interaction_matrix = csr_matrix(
    (
        user_data["result_of_activity"],
        (user_data["user_id_encoded"], user_data["model_id_encoded"]),
    ),
    shape=(len(user_encoder.classes_), len(model_encoder.classes_)),
)

# Применение SVD для генерации признаков моделей
svd = TruncatedSVD(n_components=20, random_state=42)
model_features = svd.fit_transform(interaction_matrix)


@app.route("/recommend/similarity", methods=["POST"])
def similarity():
    model_id = request.json["model_id"]
    conn = get_db_connection()
    cars_data = pd.read_sql_query("SELECT * FROM cars", conn)
    processed_data, _ = preprocess_data(cars_data)
    similarity_matrix = calculate_similarity(processed_data)
    recommendations = find_similar_cars(model_id, processed_data, similarity_matrix)
    conn.close()
    return list(recommendations[["MODEL_ID"]]["MODEL_ID"])


@app.route("/recommend/collaborative", methods=["POST"])
def collaborative():
    user_id = request.json["user_id"]
    recommendations = recommend_colaborative_models(
        user_id,
        user_encoder,
        interaction_matrix,
        model_similarity,
        model_encoder,
    )
    return list(recommendations)


@app.route("/recommend/contentbased", methods=["POST"])
def contentbased():
    user_id = request.json["user_id"]
    user_index = user_encoder.transform([user_id])[0]
    user_preferences = model_features[user_index, :]

    # Расчёт сходства между предпочтениями пользователя и всеми моделями
    similarity_scores = cosine_similarity([user_preferences], model_features)

    # Получение рекомендаций
    recommended_model_indices = similarity_scores.argsort()[0][::-1][:5]
    # Добавляем проверку на допустимость индексов
    valid_indices = [
        idx for idx in recommended_model_indices if idx < len(model_encoder.classes_)
    ]
    recommended_models = model_encoder.inverse_transform(valid_indices)
    return list(recommended_models)


if __name__ == "__main__":
    app.run(debug=True)
