# ML-2: Логістична регресія та федеративне навчання

Ноутбук: `lab2_logistic_federated.ipynb`

Датасет: Dry Bean Dataset (UCI Machine Learning Repository, також доступний на Kaggle),
файл `data/dry_bean.csv`: 13611 зерен, 16 числових ознак, 7 класів.

## Запуск

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook lab2_logistic_federated.ipynb
```

Softmax-регресія, функція втрат, градієнти, міні-батчевий градієнтний спуск та агрегація FedAvg
реалізовані самостійно на numpy. scikit-learn використовується лише для стратифікованого розбиття,
StandardScaler, ANOVA F-тесту та метрик.
