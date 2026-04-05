import streamlit as st
import pandas as pd
from surprise import dump

st.set_page_config(page_title="Recommender systems", page_icon="🍿")

st.title("🎬 AI Прогнози vs Реални Оценки")
st.markdown("Сравнение на случайни записи от базата данни с предсказанията на SVD модела.")


# 1. Зареждане на модела и данните
@st.cache_resource
def load_all():
    try:
        _, model = dump.load('model.pkl')
        raw_data = pd.read_csv('database/ratings.csv')
        return model, raw_data
    except:
        return None, None


model, raw_data = load_all()

if model is None or raw_data is None:
    st.error("❌ Липсва 'model.pkl' или 'ratings.csv'!")
    st.stop()

# 2. Вход за потребител
user_id = st.sidebar.number_input("Избери User ID:", min_value=1, value=1, step=1)

# 3. Логика за случайните 10 филма
st.subheader(f"📊 Сравнителен анализ за Потребител {user_id}")

# Вземаме всички оценки на този потребител
user_ratings = raw_data[raw_data['userId'] == user_id]

if not user_ratings.empty:
    # Вземаме 10 случайни реда (или всички, ако са по-малко от 10)
    n_samples = min(len(user_ratings), 10)
    random_samples = user_ratings.sample(n=n_samples)

    comparison_list = []

    for _, row in random_samples.iterrows():
        m_id = int(row['movieId'])
        real_r = row['rating']

        # AI прогноза
        pred = model.predict(user_id, m_id)

        comparison_list.append({
            "Movie ID": m_id,
            "Реална Оценка (База Данни)": real_r,
            "AI Прогноза (SVD)": round(pred.est, 2),
            "Разлика": round(abs(real_r - pred.est), 2)
        })

    # Визуализация в таблица
    comparison_df = pd.DataFrame(comparison_list)
    st.table(comparison_df)

    # Средна разлика за тези 10 филма
    avg_diff = comparison_df['Разлика'].mean()
    st.info(f"💡 Средната грешка за тази случайна извадка е: **{avg_diff:.2f}**")
else:
    st.warning("Няма данни за този потребител.")

st.divider()
st.caption("Забележка: Разлики около 0.8 - 1.2 са нормални за SVD алгоритъма.")