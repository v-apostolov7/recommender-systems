# Movie Recommender System 🎬

An intelligent movie recommendation engine built with Python, utilizing Matrix Factorization (SVD) and Collaborative Filtering.

## 🚀 Live Demo
You can access the deployed application here: 
[https://recommender-systems-va-kk.streamlit.app/](https://recommender-systems-va-kk.streamlit.app/)

## 📌 Overview
This project demonstrates the implementation of a recommendation system using the **MovieLens** dataset. It compares different algorithmic approaches to predict user ratings and provides an interactive interface for real-time validation.

### Key Features:
* **Algorithm Comparison**: Evaluation of Random, KNN, and SVD models using RMSE.
* **SVD Implementation**: High-accuracy predictions based on Latent Factor analysis (Netflix Prize standard).
* **Interactive UI**: Built with Streamlit to visualize predictions vs. actual historical data.
* **Scalable Design**: Optimized to handle large datasets through efficient sampling and serialization.

## 📁 Project Structure
```text
├── main.py                # Model training and evaluation script
├── app.py                 # Streamlit web interface
├── ratings.csv            # Dataset (sampled from MovieLens 32M)
├── model.pkl              # Serialized SVD model
├── requirements.txt       # Project dependencies
└── src/                   # Source code modules (data_loader, models, train_eval)

🛠️ Installation & Local Setup
Clone the repository (or copy the project folder).

Create a virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

Bash
pip install -r requirements.txt
Run the training script (to generate model.pkl):

Bash
python main.py
Launch the Streamlit app:

Bash
streamlit run app.py
📊 Results
The SVD model achieved a Root Mean Squared Error (RMSE) of ~0.98, significantly outperforming baseline models. The system successfully captures latent user preferences even in highly sparse datasets.

📚 Technologies Used
Python 3.x

Pandas: Data manipulation

Scikit-Surprise: Machine Learning & Recommendation logic

Streamlit: Web deployment and UI

PyCharm: Development IDE


---

### Какво да направиш сега:
1. Отвори **PyCharm**.
2. Десен бутон върху основната папка на проекта -> `New` -> `File`.
3. Кръсти го точно `README.md`.
4. Постави горния текст вътре и го запиши.

**Съвет за защитата:**
Когато показваш линка в Streamlit Cloud, кажи: 
> *"I have deployed the model to a cloud environment using Streamlit Community Cloud, which allows for cross-platform accessibility and real-time demonstration of the SVD algorithm's performance."*



Вече си готов на 100%! Има ли нещо друго, което искаш да добавим или променим?
> **Note on Dataset:** The original `database/ratings.csv` is not included in the cloud repository due to its large size (>100MB). To run the project locally, you must download the MovieLens Latest dataset and place it in the `database/` folder.
