import streamlit as st
import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================
# 1. Завантаження ресурсів NLTK
# ==========================================
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# ==========================================
# 2. Функція очищення тексту
# ==========================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Жодного NLTK чи стоп-слів. Просто залишаємо текст як є (без ком) і робимо маленьким.
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()


# ==========================================
# 3. Завантаження моделей та ініціалізація стану
# ==========================================
@st.cache_resource
def load_models():
    # Переконайтеся, що ви завантажили нові версії .pkl після перенавчання (10k features)
    model = joblib.load('bug_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer


try:
    model, vectorizer = load_models()
except FileNotFoundError:
    st.error("Помилка: Не знайдено файли моделі. Переконайтеся, що .pkl файли лежать у тій самій папці.")
    st.stop()

if 'bug_history' not in st.session_state:
    st.session_state.bug_history = []

# ==========================================
# 4. НАВІГАЦІЯ ТА ІНТЕРФЕЙС
# ==========================================
st.set_page_config(page_title="Класифікатор багів", page_icon="🛠", layout="wide")

st.sidebar.title("Навігація")
page = st.sidebar.radio("Оберіть розділ:", ["📝 Нове звернення", "📊 Таблиця категорій (Історія)"])

if page == "📝 Нове звернення":
    st.title("🛠 Інтелектуальна система класифікації звернень")
    st.markdown("""
    Цей модуль використовує алгоритм **Random Forest** та методи **NLP** для автоматичного визначення категорії баг-репорту.
    """)

    # --- МІЙ ВСТАВЛЕНИЙ ШМАТОК: ГАЙД ДЛЯ КОРИСТУВАЧА ---
    with st.expander("ℹ️ Як написати репорт для 90%+ точності?"):
        st.markdown("""
        Для того, щоб ШІ максимально точно визначив категорію, дотримуйтесь структури:
        1. **Що зламалося?** (напр. *Database, Login page, Jenkins pipeline*)
        2. **Яка помилка?** (напр. *Timeout, Connection failed, Error 500*)
        3. **Контекст/Технологія:** (напр. *Docker, AWS, PostgreSQL, React*)

        **Приклад ідеального запиту:** `Connection timeout in PostgreSQL database cluster. Error 503.`
        """)
    # ------------------------------------------------

    user_input = st.text_area(
        "Опис проблеми (англійською):",
        height=150,
        placeholder="Введіть опис проблеми, наприклад: API request failed with timeout in Docker container..."
    )

    if st.button("Класифікувати звернення"):
        if user_input.strip():
            with st.spinner('🤖 Аналізую текст...'):
                cleaned_input = clean_text(user_input)
                vectorized_input = vectorizer.transform([cleaned_input])

                prediction = model.predict(vectorized_input)[0]
                probabilities = model.predict_proba(vectorized_input)[0]
                max_prob = max(probabilities) * 100

                st.session_state.bug_history.append({
                    "Оригінальний текст": user_input,
                    "Передбачена категорія": prediction,
                    "Впевненість моделі (%)": round(max_prob, 1)
                })

                st.success(f"**Категорія:** {prediction}")

                if max_prob > 80:
                    st.info(f"**Впевненість моделі:** {max_prob:.1f}%")
                else:
                    st.warning(f"**Низька впевненість:** {max_prob:.1f}% (Потребує перевірки модератором)")
        else:
            st.warning("Будь ласка, введіть текст звернення.")

elif page == "📊 Таблиця категорій (Історія)":
    st.title("📊 Історія оброблених звернень")

    if len(st.session_state.bug_history) > 0:
        df_history = pd.DataFrame(st.session_state.bug_history)
        categories = ["Всі"] + list(df_history["Передбачена категорія"].unique())
        selected_category = st.selectbox("Фільтр за категорією:", categories)

        if selected_category != "Всі":
            df_history = df_history[df_history["Передбачена категорія"] == selected_category]

        st.dataframe(df_history, use_container_width=True, hide_index=True)

        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Завантажити звіт (CSV)",
            data=csv,
            file_name='bug_reports_history.csv',
            mime='text/csv',
        )
    else:
        st.info("Історія порожня. Перейдіть у розділ «Нове звернення» та класифікуйте кілька репортів.")
