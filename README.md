# The Retention Architect

Проект предсказывает `churn_status` пользователя в 3 класса:

- `not_churned` — пользователь не отвалится
- `vol_churn` — пользователь уйдет сам
- `invol_churn` — пользователь потеряется из-за проблем с оплатой

Главная идея проекта: предсказать не только сам факт оттока, но и его причину. Это важно для бизнеса, потому что разные типы оттока требуют разных действий:

- `vol_churn` лечится продуктом, retention-механиками, реактивацией, контентом
- `invol_churn` лечится платежным флоу, retry logic, dunning, wallet/payment methods

## Текущие результаты

Модель сейчас обучается на уменьшенном основном датасете, который лежит прямо в репозитории.

| Метрика | Значение |
|---|---|
| Macro F1 (OOF) | **0.4120** |
| Accuracy | **0.53** |
| Train users | **1,500** |
| Test users | **1,000** |
| Train generations rows | **499,189** |

## Какие файлы за что отвечают

### Код

- [retention_pipeline.py](/Users/agybay/Downloads/Higgsfield-DS-main/retention_pipeline.py)
  Главный модуль пайплайна.
  Отвечает за:
  - чтение данных
  - feature engineering
  - сборку train/test таблиц
  - кодирование признаков
  - подготовку `X_train`, `X_test`, `y_train`

- [run_model.py](/Users/agybay/Downloads/Higgsfield-DS-main/run_model.py)
  Точка входа для запуска модели.
  Отвечает за:
  - вызов подготовки данных из `retention_pipeline.py`
  - обучение `RandomForest`
  - кросс-валидацию
  - сохранение `submission.csv` и `submission_detailed.csv`

- [README.md](/Users/agybay/Downloads/Higgsfield-DS-main/README.md)
  Документация проекта.

### Данные

- [Train Data/train_users.csv](/Users/agybay/Downloads/Higgsfield-DS-main/Train%20Data/train_users.csv)
  Главная train-таблица: `user_id` и целевой класс `churn_status`.

- [Train Data/train_users_properties.csv](/Users/agybay/Downloads/Higgsfield-DS-main/Train%20Data/train_users_properties.csv)
  Данные о подписке и стране.

- [Train Data/train_users_purchases.csv](/Users/agybay/Downloads/Higgsfield-DS-main/Train%20Data/train_users_purchases.csv)
  История покупок пользователя.

- [Train Data/train_users_transaction_attempts.csv](/Users/agybay/Downloads/Higgsfield-DS-main/Train%20Data/train_users_transaction_attempts.csv)
  Попытки транзакций, включая неуспешные.

- [Train Data/train_users_quizzes.csv](/Users/agybay/Downloads/Higgsfield-DS-main/Train%20Data/train_users_quizzes.csv)
  Онбординг / анкетные сигналы.

- [train_users_generations.csv/train_users_generations.csv](/Users/agybay/Downloads/Higgsfield-DS-main/train_users_generations.csv/train_users_generations.csv)
  Самый важный поведенческий лог.
  Здесь хранятся события генераций, из которых строятся usage-trends, burst-patterns, retention-метрики, recency и активность по дням.

- [Test Data/test_users.csv](/Users/agybay/Downloads/Higgsfield-DS-main/Test%20Data/test_users.csv)
  Тестовые пользователи без таргета.

- [Test Data/test_users_properties.csv](/Users/agybay/Downloads/Higgsfield-DS-main/Test%20Data/test_users_properties.csv)
- [Test Data/test_users_purchases.csv](/Users/agybay/Downloads/Higgsfield-DS-main/Test%20Data/test_users_purchases.csv)
- [Test Data/test_users_transaction_attempts.csv](/Users/agybay/Downloads/Higgsfield-DS-main/Test%20Data/test_users_transaction_attempts.csv)
- [Test Data/test_users_quizzes.csv](/Users/agybay/Downloads/Higgsfield-DS-main/Test%20Data/test_users_quizzes.csv)
- [Test Data/test_users_generations.csv](/Users/agybay/Downloads/Higgsfield-DS-main/Test%20Data/test_users_generations.csv)
  Те же сущности, но для теста.

### Выходы

- [submission.csv](/Users/agybay/Downloads/Higgsfield-DS-main/submission.csv)
  Финальный файл предсказаний:
  `user_id -> churn_status`

- [submission_detailed.csv](/Users/agybay/Downloads/Higgsfield-DS-main/submission_detailed.csv)
  Расширенный вывод:
  - `predicted_class`
  - вероятность каждого класса
  - `confidence`

## Как работает пайплайн

### 1. Загрузка данных

В [retention_pipeline.py](/Users/agybay/Downloads/Higgsfield-DS-main/retention_pipeline.py) функция `load_raw_data()` читает все train/test CSV из корня проекта.

На этом шаге проект получает:

- train-пользователей с таргетом
- test-пользователей без таргета
- дополнительные таблицы с properties / purchases / transactions / quizzes / generations

### 2. Feature engineering по каждой таблице

Проект не учится на “сырых” таблицах напрямую. Сначала каждая таблица превращается в user-level признаки.

#### Properties

Функция: `build_properties_features()`

Что делает:

- парсит `subscription_start_date`
- извлекает месяц подписки
- вычисляет день недели
- вытаскивает час старта
- считает относительный ранг даты регистрации
- добавляет частоту страны (`country_freq`)

Смысл:
- страна и время старта подписки могут коррелировать и с поведением, и с платежной инфраструктурой

#### Purchases

Функция: `build_purchase_features()`

Что делает:

- считает количество покупок
- суммарную выручку
- средний / max / min чек
- число типов покупок
- количество покупок каждого типа
- расстояние между первой и последней покупкой

Смысл:
- это признаки монетизации и зрелости пользователя

#### Transaction attempts

Функция: `build_transaction_features()`

Это одна из самых важных частей проекта.

Что делает:

1. Делит транзакции на успешные и неуспешные
2. Для успешных считает card-risk признаки:
   - prepaid
   - virtual
   - business
   - 3D secure
   - digital wallet
   - cvc pass rate
   - country mismatch
   - средний и максимальный платеж
3. Для failed transaction attempts строит fingerprint:
   - `card_brand`
   - `card_country`
   - `card_funding`
   - `bank_name`
   - `billing_address_country`
   - `is_prepaid`
   - `is_virtual`
4. Через fingerprint сопоставляет failed транзакции пользователям, даже если в failed tx нет `user_id`
5. Считает weighted fail count и breakdown по failure code

Смысл:
- это ядро для предсказания `invol_churn`
- модель учится распознавать пользователей, у которых повышен риск потери из-за проблем с картой или оплатой

#### Generations

Функции:

- `agg_gen_chunk()`
- `finalize_gen_features()`
- `build_generation_features_chunked()`
- `build_generation_features()`

Это самый сильный блок поведенческих фич.

Что делает:

- читает train generations чанками, чтобы не расходовать много памяти
- считает число генераций, completed/failed/nsfw/canceled
- считает кредиты, длительности, типы генераций
- строит активность по дням `active_day_0 ... active_day_13`
- строит usage-trend между week1 и week2
- строит `days_since_last_gen`
- строит burst-признаки:
  - `day01_ratio`
  - `day01_credits_ratio`
  - `burst_flag`
- строит short-life признаки:
  - `is_short_life`
  - `is_one_day_user`
  - `first_quit_day`
  - `max_active_streak`

Смысл:
- это ядро для предсказания `vol_churn`
- если пользователь резко вспыхнул, быстро погенерил, а потом исчез, это очень похоже на voluntary churn

#### Quizzes

Функция: `build_quiz_features()`

Что делает:

- берет `source`, `experience`, `usage_plan`, `frustration`, `first_feature`, `role`, `team_size`
- преобразует их в чистые user-level признаки
- отмечает наличие quiz-данных

Смысл:
- это сигналы онбординга и намерений пользователя

### 3. Объединение всех признаков

Функция: `merge_all_features()`

На этом шаге всё сводится в одну таблицу на пользователя.

Дополнительно здесь создаются interaction features:

- `credits_x_failure_rate`
- `trend_x_recency`
- `spend_per_gen`
- `gen_total_x_purchases`
- `prepaid_x_failure`
- `prepaid_x_cvc`
- `payment_risk_score`
- `burst_x_short_life`
- `high_usage_short_life`
- `credits_per_active_day`
- `abrupt_decline`
- `sustained_engagement`
- `exploration_score`
- `purchase_engagement`

Смысл:
- это признаки, которые лучше отражают не отдельные факты, а комбинации поведения

### 4. Очистка признаков

Функция: `clean_features()`

Что делает:

- clip extreme outliers
- добавляет `log1p` версии сильно скошенных числовых признаков

Смысл:
- стабилизировать распределения и уменьшить влияние хвостов

### 5. Поведенческий кластер

Функция: `add_cluster_features()`

Что делает:

- берет набор behavioral features
- масштабирует их
- обучает `KMeans`
- добавляет `cluster_id`

Смысл:
- это компактный сигнал типа поведения пользователя

### 6. Подготовка матрицы для модели

Что происходит дальше:

- выравниваются колонки train/test
- кодируется таргет через `LabelEncoder`
- categorical features приводятся к строкам
- убирается `subscription_plan`
- категориальные признаки кодируются
- для `country_code` делается target encoding по классам

Итог:

- `X_train`
- `X_test`
- `X_train_encoded`
- `X_test_encoded`
- `y_train`

Все они собираются в объект `PreparedModelData`.

## Как обучается модель

Обучение происходит в [run_model.py](/Users/agybay/Downloads/Higgsfield-DS-main/run_model.py).

Текущая основная модель:

- `RandomForestClassifier`
- `n_estimators=300`
- `min_samples_leaf=2`
- `class_weight='balanced_subsample'`
- `n_jobs=1`

Схема обучения:

1. Берутся подготовленные признаки из `prepare_model_data()`
2. Строится `StratifiedKFold`
3. На каждом фолде:
   - модель учится на train-fold
   - предсказывает вероятности на validation-fold
   - предсказывает вероятности на test
4. Из OOF-предсказаний считается итоговый `Macro F1`
5. Вероятности на test усредняются по фолдам
6. Класс выбирается через `argmax(probabilities)`

Почему `StratifiedKFold`:

- потому что классы несбалансированы 50 / 25 / 25
- стратификация сохраняет это распределение в каждом фолде

Почему `Macro F1`:

- потому что важно хорошо различать все 3 класса, а не просто угадывать самый частый `not_churned`

## Что лежит в `submission_detailed.csv`

Файл содержит:

- `user_id`
- `predicted_class`
- `prob_invol_churn`
- `prob_not_churned`
- `prob_vol_churn`
- `confidence`

Зачем он нужен:

- можно смотреть не только класс, но и уверенность
- можно вручную анализировать спорные кейсы
- можно использовать вероятности для downstream бизнес-логики

## Как объяснить проект другим

### В одном предложении

“Это модель, которая по поведению пользователя и платежным сигналам предсказывает не только отток, но и его тип: добровольный или связанный с оплатой.”

### За 30 секунд

“Мы собрали user-level признаки из нескольких таблиц: подписка, покупки, платежи, генерации и онбординг. Самые сильные сигналы идут из темпа использования продукта и риска оплаты. Дальше обучаем многоклассовую модель, которая делит пользователей на `not_churned`, `vol_churn` и `invol_churn`. Это помогает понимать, нужно ли чинить retention или billing.”

### Для технаря

“Это multiclass tabular pipeline. Feature engineering собирает поведенческие, платежные и onboarding признаки на уровне пользователя. В train используется stratified CV, categorical encoding и class probabilities. Основная runnable-модель сейчас — CPU-friendly RandomForest на reduced dataset.”

## Как запускать

Из корня проекта:

```bash
.venv/bin/python run_model.py
```

После запуска обновятся:

- [submission.csv](/Users/agybay/Downloads/Higgsfield-DS-main/submission.csv)
- [submission_detailed.csv](/Users/agybay/Downloads/Higgsfield-DS-main/submission_detailed.csv)

## Что оставлено в проекте

Сейчас в проекте оставлены только реально нужные вещи:

- данные
- основной модуль пайплайна
- основной скрипт обучения
- результаты
- документация

Удалено:

- `__pycache__`
- старые временные артефакты
- неиспользуемый notebook-runtime путь

## Итог

Проект сейчас устроен так:

1. [retention_pipeline.py](/Users/agybay/Downloads/Higgsfield-DS-main/retention_pipeline.py) — готовит признаки
2. [run_model.py](/Users/agybay/Downloads/Higgsfield-DS-main/run_model.py) — обучает модель
3. [submission.csv](/Users/agybay/Downloads/Higgsfield-DS-main/submission.csv) и [submission_detailed.csv](/Users/agybay/Downloads/Higgsfield-DS-main/submission_detailed.csv) — результат

То есть архитектура теперь простая, нормальная и “по-питоновски правильная”: без `exec`, без запуска кода из `.ipynb`, с обычными модулями и импортами.
