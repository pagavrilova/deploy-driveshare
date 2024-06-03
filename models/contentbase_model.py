import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Для создания системы контентной фильтрации на основе предоставленных данных (где основной информацией является лишь идентификатор модели и взаимодействия пользователя с ней), нам необходимо придумать способ описания характеристик каждой модели, используя доступные данные. Поскольку в исходных данных нет явных характеристик моделей, мы можем использовать информацию о поведении пользователей для выделения скрытых признаков моделей. Это можно сделать с помощью методов матричной факторизации, таких как Singular Value Decomposition (SVD) или Non-negative Matrix Factorization (NMF), которые генерируют признаки на основе взаимодействий пользователей с моделями.

# Этапы реализации:
# Создание матрицы взаимодействий: Используем матрицу, созданную ранее.
# Применение матричной факторизации: Используем SVD или NMF для генерации признаков моделей на основе взаимодействий.
# Расчёт сходства: Используем косинусное сходство между векторами признаков моделей.
# Генерация рекомендаций: Для данного пользователя находим модели с наибольшим сходством к моделям, которые ему нравятся.

if __name__ == "__main__":
    # Загрузка данных
    data = pd.read_csv("./data_gen/User Data.csv")

    # Кодирование идентификаторов пользователей и моделей для использования в матрице
    user_encoder = LabelEncoder()
    model_encoder = LabelEncoder()

    data["user_id_encoded"] = user_encoder.fit_transform(data["user_id"])
    data["model_id_encoded"] = model_encoder.fit_transform(data["model_id"])

    # Создание разреженной матрицы взаимодействий
    interaction_matrix = csr_matrix(
        (
            data["result_of_activity"],
            (data["user_id_encoded"], data["model_id_encoded"]),
        ),
        shape=(len(user_encoder.classes_), len(model_encoder.classes_)),
    )

    # Применение SVD для генерации признаков моделей
    svd = TruncatedSVD(n_components=20, random_state=42)
    model_features = svd.fit_transform(interaction_matrix)

    # Выбор конкретного пользователя для рекомендаций
    specific_user_id = (
        "053013cb46c1679053157f55eb41b49145c2c20cacf4f7ada3619c628a018023"
    )
    if specific_user_id in data["user_id"].values:
        user_index = user_encoder.transform([specific_user_id])[0]
        user_preferences = model_features[user_index, :]

        # Расчёт сходства между предпочтениями пользователя и всеми моделями
        similarity_scores = cosine_similarity([user_preferences], model_features)

        # Получение рекомендаций
        recommended_model_indices = similarity_scores.argsort()[0][::-1][:5]
        # Добавляем проверку на допустимость индексов
        valid_indices = [
            idx
            for idx in recommended_model_indices
            if idx < len(model_encoder.classes_)
        ]
        recommended_models = model_encoder.inverse_transform(valid_indices)

        print("Рекомендованные модели:", recommended_models)
    else:
        print("Пользователь не найден в базе данных.")
