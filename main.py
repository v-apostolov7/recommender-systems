from src.data_loader import load_and_prepare_data
from src.models import get_models
from src.train_eval import evaluate_models
from surprise import SVD, dump


def main():
    path = 'database/ratings.csv'

    # 1. Зареждане
    print("Зареждане на данни...")
    data = load_and_prepare_data(path)

    # 2. Оценка на алгоритмите (за т. 1.4 от документацията)
    all_models = get_models()
    print("Стартиране на оценка (RMSE)...")
    performance = evaluate_models(data, all_models)

    print("\n--- РЕЗУЛТАТИ ЗА ДОКУМЕНТАЦИЯТА ---")
    for name, rmse in performance.items():
        print(f"{name}: RMSE = {rmse:.4f}")

    # 3. Обучение на финалния модел и запис за Streamlit
    print("\nЗаписване на SVD модел в model.pkl...")
    full_trainset = data.build_full_trainset()
    final_algo = SVD()
    final_algo.fit(full_trainset)
    dump.dump('model.pkl', algo=final_algo)
    print("✅ Готово!")


if __name__ == "__main__":
    main()