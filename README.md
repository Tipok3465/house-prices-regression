# House Prices Regression

## Цель проекта

Цель проекта -- построить полный ML-пайплайн для задачи регрессии из соревнования Kaggle **House Prices: Advanced Regression Techniques**.

Задача состоит в том, чтобы предсказать итоговую стоимость продажи дома (`SalePrice`) на основе числовых и категориальных признаков, описывающих объект недвижимости.

Проект делается как pet project с акцентом на:

- разведочный анализ данных;
- корректную обработку признаков;
- построение baseline-модели;
- отслеживание экспериментов;
- улучшение качества модели через feature engineering;
- сохранение лучшей модели;
- генерацию submission-файла для Kaggle.

---

## Датасет

Данные взяты из соревнования Kaggle **House Prices: Advanced Regression Techniques**.

Датасет содержит информацию о домах в городе Ames, Iowa. Для каждого дома доступны признаки, описывающие:

- общее качество дома;
- жилую площадь;
- характеристики подвала;
- характеристики гаража;
- район;
- год постройки и ремонта;
- условия продажи.

Датасеты проекта:

```text
data/raw/train.csv
data/raw/test.csv
```

Целевая переменная:

```text
SalePrice
```

Файлы `train.csv` и `test.csv` не изменяются вручную. Все преобразования выполняются в ноутбуках или в коде из папки `src/`.

---

## Метрика

В соревновании используется **RMSE между логарифмом предсказанной цены и логарифмом реальной цены**.

Поэтому во время обучения целевая переменная преобразуется:

```python
y = np.log1p(train["SalePrice"])
```

После предсказания значения переводятся обратно в исходный масштаб:

```python
predictions = np.expm1(predictions_log)
```

Такое преобразование помогает:

- уменьшить влияние очень дорогих домов;
- сделать распределение целевой переменной ближе к нормальному;
- оптимизировать модель под метрику соревнования;
- лучше учитывать относительные ошибки, а не только абсолютную разницу в цене.

---

## Текущий результат

Текущий лучший результат на Kaggle Public Leaderboard: **0.13192**.

Этот результат получен с помощью простого ансамбля:

```text
0.5 * Ridge tuned + 0.5 * Lasso tuned
```

- baseline Ridge: `0.13375`;
- лучший текущий Ridge после feature engineering: `0.13261`;
- ансамбль Ridge + Lasso: `0.13192`.

На текущем этапе самый заметный прирост дал не более сложный алгоритм, а простое усреднение двух регуляризованных линейных моделей.

---

## Пайплайн проекта

Текущий пайплайн:

```text
1. Загрузка train/test данных
2. Разведочный анализ данных
3. Анализ распределения SalePrice
4. Анализ пропусков
5. Анализ корреляций
6. Построение baseline preprocessing pipeline
7. Обучение baseline Ridge Regression
8. Feature engineering
9. Проверка влияния удаления выбросов
10. Подбор alpha для Ridge Regression
11. Сравнение sklearn-моделей
12. Tuning линейных моделей
13. Отдельный preprocessing для tree-based моделей
14. Простой ансамбль Ridge + Lasso
15. Генерация submission-файлов
16. Сохранение результатов экспериментов
```

Планируемые следующие шаги:

```text
1. Сделать единый train.py для воспроизводимого обучения
2. Сравнить несколько моделей
3. Подобрать гиперпараметры для лучших моделей
4. Сохранить лучшую модель в models/
```

---

## Структура репозитория

```text
house-price-regression/
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── data_description.txt
│   │
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_tuning_and_ensembling.ipynb
│
├── reports/
│   ├── figures/
│   └── model_results.csv
│
├── src/
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   └── utils.py
│
├── submissions/
│   ├── submission_baseline.csv
│   ├── submission_feature_engineering.csv
│   ├── submission_ridge_tuned.csv
│   ├── submission_5_ridge.csv
│   ├── submission_6_lasso.csv
│   ├── submission_7_elasticnet.csv
│   ├── submission_8_random_forest.csv
│   ├── submission_10_gradient_boosting.csv
│   ├── submission_011_ridge_tuned.csv
│   ├── submission_012_lasso_tuned.csv
│   ├── submission_013_elasticnet_tuned.csv
│   ├── submission_014_random_forest_tree_prep.csv
│   ├── submission_015_gradient_boosting_tree_prep.csv
│   └── submission_017_ridge_lasso_ensemble.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Baseline-подход

Первая модель проекта -- **Ridge Regression**.

Ridge была выбрана как baseline-модель, потому что:

- она простая и быстро обучается;
- хорошо подходит для первой проверки пайплайна;
- работает с большим количеством one-hot encoded признаков;
- использует регуляризацию, которая помогает бороться с переобучением;
- даёт понятную стартовую точку для дальнейших улучшений.

Preprocessing реализован через `Pipeline` и `ColumnTransformer`.

Для числовых признаков используется:

```text
SimpleImputer(strategy="median")
StandardScaler()
```

Для категориальных признаков используется:

```text
SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore")
```

На baseline-этапе preprocessing намеренно сделан простым. Необходимо получить первый корректный результат и проверить, что весь пайплайн работает от загрузки данных до submission-файла.


---

## Feature engineering

Feature engineering вынесен в `src/features.py`.

Основные изменения:

- часть пропусков заменяется на `"None"`, если пропуск означает отсутствие объекта;
- добавлены новые признаки:
  - `TotalSF`;
  - `TotalBath`;
  - `HouseAge`;
  - `RemodAge`;
  - `HasGarage`;
  - `HasBasement`;
  - `HasFireplace`;
- отдельно проверяется гипотеза об удалении выбросов;
- train и test проходят одинаковую обработку через общие функции.

---

## Результаты экспериментов

| Experiment | Model | CV RMSE | CV std | Kaggle score | Notes |
|---|---|---:|---:|---:|---|
| 001 | Ridge | 0.14679 | 0.03932 | 0.13375 | baseline |
| 002 | Ridge | 0.11489 | 0.00818 | 0.13435 | feature engineering + outlier removal |
| 003 | Ridge | 0.14711 | 0.03976 | 0.13360 | feature engineering without outlier removal |
| 004 | Ridge tuned | 0.14700 | 0.03976 | 0.13262 | feature engineering + tuned alpha without outlier removal |
| 005 | Ridge | 0.14702 | 0.04052 | 0.13261 | model comparison with feature engineering |
| 006 | Lasso | 0.14171 | 0.04326 | 0.13277 | model comparison with feature engineering |
| 007 | ElasticNet | 0.14300 | 0.04132 | 0.13429 | model comparison with feature engineering |
| 008 | RandomForestRegressor | 0.14206 | 0.01889 | 0.14325 | model comparison with feature engineering |
| 009 | HistGradientBoostingRegressor | 0.13327 | 0.01670 | - | model comparison with feature engineering |
| 010 | GradientBoostingRegressor | 0.13387 | 0.01760 | 0.13680 | model comparison with feature engineering |
| 011 | Ridge tuned | - | - | 0.13261 | tuned linear model |
| 012 | Lasso tuned | - | - | 0.13277 | tuned linear model |
| 013 | ElasticNet tuned | - | - | 0.13286 | tuned linear model |
| 014 | RandomForestRegressor tree-prep | - | - | 0.14289 | tree-specific preprocessing |
| 015 | GradientBoostingRegressor tree-prep | - | - | 0.13395 | tree-specific preprocessing |
| 017 | Ridge + Lasso ensemble | - | - | 0.13192 | 0.5 Ridge tuned + 0.5 Lasso tuned |

Результаты также сохраняются в файл:

```text
reports/model_results.csv
---

## Основные наблюдения

- Удаление выбросов сильно улучшило локальный CV score, но ухудшило результат на Kaggle.
- Feature engineering без удаления выбросов немного улучшил результат на Public Leaderboard.
- Подбор `alpha` для Ridge Regression дал лучший результат на этапе feature engineering, но локальный CV почти не изменился.
- В сравнении sklearn-моделей лучший Kaggle score снова показала Ridge Regression.
- Lasso дал близкий результат, но немного уступил Ridge.
- RandomForestRegressor и GradientBoostingRegressor не улучшили результат без дополнительной настройки.
- HistGradientBoostingRegressor показал лучший локальный CV RMSE, но был временно исключён из submission-цикла из-за слишком долгого финального обучения в текущем pipeline.
- Улучшение на локальной cross-validation не всегда гарантирует улучшение на скрытой test-выборке Kaggle.

---

## Как запустить проект

1. Установить зависимости:

```bash
pip install -r requirements.txt
```

2. Положить файлы соревнования в `data/raw/`:

```text
train.csv
test.csv
data_description.txt
```

3. Запустить ноутбуки по порядку:

```text
notebooks/01_eda.ipynb
notebooks/02_baseline.ipynb
notebooks/03_feature_engineering.ipynb
notebooks/04_modeling.ipynb
notebooks/05_tuning_and_ensembling.ipynb
```

4. После запуска submission-файлы будут сохранены в папке ```submissions/```

---

## Следующие шаги

На текущем этапе лучший результат показывает ансамбль Ridge + Lasso.

Дальше можно двигаться в нескольких направлениях.

### 1. Улучшение линейных моделей

- расширить сетки параметров для Ridge, Lasso и ElasticNet;
- попробовать разные веса ансамбля Ridge + Lasso;
- проверить ансамбль Ridge + Lasso + ElasticNet;
- сделать отбор признаков;
- проверить устойчивость результата на разных validation splits.

### 2. Отдельная настройка бустингов

- попробовать XGBoost;
- попробовать LightGBM;
- попробовать CatBoost;
- подобрать гиперпараметры;
- проверить другой preprocessing для категориальных признаков;
- сравнить результаты с текущим ансамблем Ridge + Lasso.

### 3. Сохранение лучшей модели

- выбрать лучшую модель по CV и Kaggle Public Leaderboard;
- обучить её на всех данных;
- сохранить в `models/`;
- создать `src/train.py` для воспроизводимого запуска всего пайплайна.

