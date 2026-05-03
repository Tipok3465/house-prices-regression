# House Prices Regression

## Цель проекта

Цель проекта — построить полный ML-пайплайн для задачи регрессии из соревнования Kaggle **House Prices: Advanced Regression Techniques**.

Задача состоит в том, чтобы предсказать итоговую стоимость продажи дома (`SalePrice`) на основе числовых и категориальных признаков, описывающих объект недвижимости.

Проект делается как pet project с акцентом на:

- разведочный анализ данных;
- корректную обработку признаков;
- построение baseline-модели;
- отслеживание экспериментов;
- улучшение качества модели через feature engineering;
- сравнение разных классов моделей;
- настройку гиперпараметров;
- простые ансамбли;
- генерацию submission-файлов для Kaggle.

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
data/raw/data_description.txt
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

Текущий лучший результат на Kaggle Public Leaderboard: **0.12360**.

Этот результат получен с помощью смешанного ансамбля:

```text
0.5 * (Ridge + Lasso ensemble) + 0.5 * (XGBoost + LightGBM + CatBoost ensemble)
```

Динамика улучшений:

- baseline Ridge: `0.13375`;
- Ridge + Lasso ensemble: `0.13192`;
- XGBoost: `0.12784`;
- CatBoost: `0.12502`;
- boosting ensemble: `0.12454`;
- linear + boosting ensemble: `0.12360`.

Самый сильный прирост дал не один отдельный алгоритм, а ансамблирование моделей разных классов.

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
15. Boosting-модели: XGBoost, LightGBM, CatBoost
16. Ансамбль boosting-моделей
17. Смешанный ансамбль линейных моделей и boosting-моделей
18. Генерация submission-файлов
19. Сохранение результатов экспериментов
```

Feature engineering, scoring, создание submission-файлов и сохранение результатов вынесены в модули:

```text
src/features.py
src/utils.py
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
│   ├── 05_tuning_and_ensembling.ipynb
│   └── 06_boosting_models.ipynb
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
│   ├── submission_005_ridge.csv
│   ├── submission_006_lasso.csv
│   ├── submission_007_elasticnet.csv
│   ├── submission_008_random_forest.csv
│   ├── submission_010_gradient_boosting.csv
│   ├── submission_011_ridge_tuned.csv
│   ├── submission_012_lasso_tuned.csv
│   ├── submission_013_elasticnet_tuned.csv
│   ├── submission_014_random_forest_tree_prep.csv
│   ├── submission_015_gradient_boosting_tree_prep.csv
│   ├── submission_017_ridge_lasso_ensemble.csv
│   ├── submission_018_xgboost.csv
│   ├── submission_019_lightgbm.csv
│   ├── submission_020_catboost.csv
│   ├── submission_021_boosting_ensemble.csv
│   └── submission_022_linear_boosting_ensemble.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Baseline-подход

Первая модель проекта — **Ridge Regression**.

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

| Experiment | Model | CV / OOF RMSE | CV std | Kaggle score | Notes |
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
| 011 | Ridge tuned | 0.14702 | 0.04052 | 0.13261 | tuned linear model |
| 012 | Lasso tuned | 0.14171 | 0.04326 | 0.13277 | tuned linear model |
| 013 | ElasticNet tuned | 0.14172 | 0.04305 | 0.13286 | tuned linear model |
| 014 | RandomForestRegressor tree-prep | 0.14160 | 0.01904 | 0.14289 | tree-specific preprocessing |
| 015 | GradientBoostingRegressor tree-prep | 0.13120 | 0.01600 | 0.13395 | tree-specific preprocessing |
| 017 | Ridge + Lasso ensemble | 0.14922 | - | 0.13192 | OOF RMSE; 0.5 Ridge + 0.5 Lasso |
| 018 | XGBoost | 0.12616 | 0.01700 | 0.12784 | boosting model with tree-specific preprocessing |
| 019 | LightGBM | 0.13108 | 0.01810 | - | boosting model with tree-specific preprocessing |
| 020 | CatBoost | 0.12345 | 0.01647 | 0.12502 | boosting model with tree-specific preprocessing |
| 021 | XGBoost + LightGBM + CatBoost ensemble | 0.12530 | - | 0.12454 | OOF RMSE; boosting ensemble |
| 022 | Linear ensemble + Boosting ensemble | 0.13535 | - | 0.12360 | OOF RMSE; mixed ensemble |

Результаты также сохраняются в файл:

```text
reports/model_results.csv
```
---

## Итоговые результаты

На текущем этапе результаты можно интерпретировать так:

- лучшая одиночная модель по локальной CV: **CatBoost** (`CV RMSE = 0.12345`);
- лучший boosting submission: **XGBoost + LightGBM + CatBoost ensemble** (`Kaggle score = 0.12454`);
- лучший Public Leaderboard submission: **Linear ensemble + Boosting ensemble** (`Kaggle score = 0.12360`);
- важная оговорка: лучший Public Leaderboard score не обязательно означает лучшую обобщающую способность модели.

Mixed ensemble дал лучший Kaggle Public Leaderboard score, но его OOF RMSE (`0.13535`) хуже, чем у CatBoost (`0.12345`) и boosting ensemble (`0.12530`).

---

## Методологические замечания

Public Leaderboard считается только на части test-выборки, поэтому результат `0.12360` нельзя считать окончательной гарантией лучшей обобщающей способности модели.

Для более надёжной оценки в ноутбуках добавлены OOF-блоки:

- OOF-оценка ансамбля Ridge + Lasso;
- OOF-оценка boosting ensemble;
- OOF-оценка mixed ensemble./

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
notebooks/06_boosting_models.ipynb
```

4. После запуска submission-файлы будут сохранены в папке:

```text
submissions/
```

5. Результаты экспериментов будут сохранены в:

```text
reports/model_results.csv
```

---

## Следующие шаги

На текущем этапе проекту больше не нужно просто добавлять новые модели. Основной следующий шаг — финализация и воспроизводимость лучшего pipeline.

1. Создать `src/train.py` для воспроизводимого запуска лучшего pipeline.
2. Сохранить лучшую модель или ансамбль в `models/`.
3. Обновить README финальным воспроизводимым запуском.
