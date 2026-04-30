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

Текущий лучший результат на Kaggle Public Leaderboard: **0.13262**.

Этот результат получен с помощью Ridge Regression, базового feature engineering и подбора параметра регуляризации `alpha`.

Улучшение относительно baseline получилось небольшим:

- baseline Ridge: `0.13375`;
- Ridge + feature engineering + tuned alpha: `0.13262`.

Это означает, что дальнейшие улучшения требуют более аккуратной валидации и сравнения разных классов моделей.

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
11. Оценка качества через cross-validation
12. Генерация submission-файлов
13. Сохранение результатов экспериментов
```

Планируемые следующие шаги:

```text
1. Вынести feature engineering в src/features.py
2. Сделать единый train.py для воспроизводимого обучения
3. Сравнить несколько моделей
4. Подобрать гиперпараметры для лучших моделей
5. Сохранить лучшую модель в models/
```

---

## Структура репозитория

```text
house-price-prediction/
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
│   └── 04_modeling.ipynb
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
│   └── submission_ridge_tuned.csv
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

## Результаты экспериментов

| Experiment | Model | CV RMSE | CV std | Kaggle score | Notes |
|---|---|---:|---:|---:|---|
| 001 | Ridge | 0.14679 | 0.03932 | 0.13375 | baseline |
| 002 | Ridge | 0.11489 | 0.00818 | 0.13435 | feature engineering + outlier removal |
| 003 | Ridge | 0.14711 | 0.03976 | 0.13360 | feature engineering without outlier removal |
| 004 | Ridge tuned | 0.14700 | 0.03976 | 0.13262 | feature engineering + tuned alpha without outlier removal |

Результаты также сохраняются в файл:

```text
reports/model_results.csv
```

---

## Основные наблюдения

- Удаление выбросов сильно улучшило локальный CV score, но ухудшило результат на Kaggle.
- Feature engineering без удаления выбросов немного улучшил результат.
- Подбор `alpha` для Ridge Regression дал лучший score, но локальный CV почти не изменился.
- Стратегию валидации нужно анализировать осторожно: улучшение на локальной cross-validation не всегда гарантирует улучшение на скрытой test-выборке.
- Следующий этап -- сравнить разные классы моделей в одинаковых условиях: линейные модели, случайный лес и градиентный бустинг.

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
```

4. После запуска submission-файлы будут сохранены в:

```text
submissions/submission_baseline.csv
submissions/submission_feature_engineering.csv
submissions/submission_ridge_tuned.csv
```

---

## Следующие шаги

Перед переходом к более сложным моделям необходимо закрыть воспроизводимость пайплайна.

Ближайшие шаги:

- вынести feature engineering в `src/features.py`;
- вынести scoring, создание submission-файлов и сохранение результатов в `src/utils.py`;
- обновить `03_feature_engineering.ipynb`, чтобы эксперименты запускались через функции из `src/`;
- создать `04_modeling.ipynb`;
- сравнить несколько классов моделей:
  - Ridge;
  - Lasso;
  - ElasticNet;
  - RandomForestRegressor;
  - HistGradientBoostingRegressor;
  - GradientBoostingRegressor;
- после этого попробовать:
  - XGBoost;
  - LightGBM.

Главная цель следующего этапа -- сравнить линейные модели, случайный лес и градиентный бустинг в одинаковых условиях.