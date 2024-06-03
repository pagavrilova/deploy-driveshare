import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Для реализации коллаборативной фильтрации, которая будет рекомендовать пользователям топ-5 интересных моделей, мы можем использовать два основных подхода:
# User-based Collaborative Filtering (Фильтрация на основе пользователя): Этот метод находит пользователей, похожих на целевого пользователя, и рекомендует модели, которые понравились похожим пользователям.
# Item-based Collaborative Filtering (Фильтрация на основе объекта): Этот метод анализирует сходство между моделями на основе оценок пользователей и рекомендует модели, похожие на те, которые пользователь уже оценил положительно.


# Функция для возвращения топ-N рекомендаций для конкретного пользователя
def recommend_colaborative_models(
    user_id,
    user_encoder,
    interaction_matrix,
    model_similarity,
    model_encoder,
    top_n=5,
):
    user_index = user_encoder.transform([user_id])[0]
    user_interactions = interaction_matrix[user_index, :]
    model_scores = user_interactions.dot(model_similarity).flatten()

    # Исключаем уже оцененные модели путем замены их оценок на минимальное значение
    model_scores[user_interactions.indices] = -1

    recommended_model_indices = model_scores.argsort()[::-1][:top_n]
    recommended_models = model_encoder.inverse_transform(recommended_model_indices)
    return recommended_models


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

    # Рассчитываем косинусное сходство между моделями
    model_similarity = cosine_similarity(
        interaction_matrix.T
    )  # Транспонируем для сравнения моделей

    # Тестируем рекомендации для случайного пользователя
    test_user_id = data["user_id"].sample(1).iloc[0]
    recommended_models = recommend_colaborative_models(
        test_user_id, interaction_matrix, model_similarity
    )

    print("Рекомендации для пользователя: ", test_user_id, ":", recommended_models)
