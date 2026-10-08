# ML-3: TF-IDF зважені ембеддинги для детекції текстів, згенерованих LLM

Ноутбук: `lab3_tfidf_embeddings.ipynb`

## Дані

У методичці рекомендовано датасет AI vs Human Text Classification Dataset з Kaggle.
Kaggle вимагає API-ключа, тому використано відкритий аналог із тією самою структурою
(колонки `text` і `generated`): `andythetechnerd03/AI-human-text` з HuggingFace.

Файл `data/ai_human_sample.csv` містить збалансовану вибірку з 6000 текстів (3000 на клас),
сформовану з повного датасету: прибрано дублікати та тексти коротші за 50 слів, seed = 42.

## Запуск

```bash
pip install numpy pandas matplotlib scikit-learn gensim spacy jupyter
python -m spacy download en_core_web_sm
jupyter notebook lab3_tfidf_embeddings.ipynb
```

Повне виконання ноутбука займає близько 2 хвилин, найдовший крок це обробка текстів у spaCy.

Примітка щодо версій: gensim потребує Python 3.12 або старішої версії, на Python 3.13+ колеса можуть бути відсутні.

## Короткий результат

Гіпотеза методички не підтвердилась: TF-IDF зважування погіршило якість для всіх трьох класифікаторів.
Найкраща комбінація це Mean Pooling плюс SVM з RBF-ядром (F1-macro 0.955, ROC-AUC 0.990).
Пояснення та числові підтвердження в розділах 6.3 і 6.4 ноутбука.
