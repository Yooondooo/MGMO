# Импорт необходимых библиотек
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Поиск CSV файлов в текущей директории
print("Поиск CSV файлов в текущей директории и поддиректориях...")
csv_files = []
for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        if filename.endswith('.csv'):
            full_path = os.path.join(dirname, filename)
            csv_files.append(full_path)
            print(f"Найден файл: {full_path}")

# Если файлы найдены, используем первый CSV файл
if csv_files:
    file_path = csv_files[0]
    print(f"\nИспользуем файл: {file_path}")
else:
    # Если файлы не найдены, запрашиваем путь у пользователя
    file_path = input("Введите путь к CSV файлу: ")
    if not os.path.exists(file_path):
        # Пробуем найти файл с похожим названием
        possible_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if possible_files:
            print(f"Найдены CSV файлы в текущей директории: {possible_files}")
            file_path = possible_files[0]
            print(f"Используем файл: {file_path}")
        else:
            # Создаем пример данных для демонстрации
            print("Файл не найден. Создаем пример данных для демонстрации...")
            np.random.seed(42)
            n_samples = 1000

            example_data = {
                'Age': np.random.randint(20, 80, n_samples),
                'BMI': np.random.uniform(18, 40, n_samples),
                'Glucose': np.random.randint(70, 200, n_samples),
                'BloodPressure': np.random.randint(60, 120, n_samples),
                'SkinThickness': np.random.randint(10, 50, n_samples),
                'Insulin': np.random.randint(0, 300, n_samples),
                'DiabetesPedigreeFunction': np.random.uniform(0.08, 2.5, n_samples),
                'Gender': np.random.choice(['Male', 'Female'], n_samples),
                'Diagnosis': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
            }

            data = pd.DataFrame(example_data)
            file_path = 'diabetes_example.csv'
            data.to_csv(file_path, index=False)
            print(f"Создан пример файла: {file_path}")

# Загрузка данных
try:
    data = pd.read_csv(file_path)
    print(f"Данные успешно загружены из: {file_path}")
except Exception as e:
    print(f"Ошибка при загрузке файла: {e}")
    print("Создаем демонстрационные данные...")

    # Создаем демонстрационные данные
    np.random.seed(42)
    n_samples = 500

    demo_data = {
        'Age': np.random.randint(20, 80, n_samples),
        'BMI': np.random.uniform(18, 40, n_samples),
        'Glucose': np.random.randint(70, 200, n_samples),
        'BloodPressure': np.random.randint(60, 120, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Diagnosis': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
    }

    data = pd.DataFrame(demo_data)

# Предварительный анализ данных
print("\n" + "=" * 50)
print("АНАЛИЗ ДАННЫХ")
print("=" * 50)

print("Размер данных:", data.shape)
print("\nПервые 5 строк:")
print(data.head())
print("\nИнформация о данных:")
print(data.info())
print("\nСтатистика данных:")
print(data.describe())
print("\nПропущенные значения:")
print(data.isnull().sum())

# Проверяем наличие целевой переменной
if 'Diagnosis' not in data.columns:
    # Если нет колонки Diagnosis, создаем ее на основе других признаков
    print("\nКолонка 'Diagnosis' не найдена. Создаем целевую переменную...")
    if 'Glucose' in data.columns and 'BMI' in data.columns:
        # Простая логика для создания целевой переменной
        data['Diagnosis'] = ((data['Glucose'] > 140) | (data['BMI'] > 30)).astype(int)
        print("Целевая переменная создана на основе Glucose и BMI")
    else:
        # Случайная целевая переменная
        np.random.seed(42)
        data['Diagnosis'] = np.random.choice([0, 1], len(data), p=[0.7, 0.3])
        print("Создана случайная целевая переменная")

# Обработка категориальных переменных
print("\n" + "=" * 50)
print("ПОДГОТОВКА ДАННЫХ")
print("=" * 50)

label_encoder = LabelEncoder()
data_encoded = data.copy()

# Кодируем категориальные колонки
categorical_columns = data_encoded.select_dtypes(include=['object']).columns

for col in categorical_columns:
    if col != 'Diagnosis':  # Целевая переменная
        print(f"Кодируем колонку: {col}")
        data_encoded[col] = label_encoder.fit_transform(data_encoded[col])
        print(f"Уникальные значения после кодирования: {data_encoded[col].unique()}")

print("\nДанные после кодирования:")
print(data_encoded.head())

# Разделение на признаки и целевую переменную
X = data_encoded.drop('Diagnosis', axis=1)
y = data_encoded['Diagnosis']

# Кодирование целевой переменной, если она категориальная
if y.dtype == 'object':
    y = label_encoder.fit_transform(y)

print(f"\nРазмер X: {X.shape}")
print(f"Размер y: {y.shape}")
print(f"Распределение классов: {np.unique(y, return_counts=True)}")

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nРазмеры выборок:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# Создание и обучение моделей
print("\n" + "=" * 50)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("=" * 50)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42)
}

best_model_name = None
best_accuracy = 0
best_model = None

results = []

for name, model in models.items():
    # Используем масштабированные данные для SVM и Logistic Regression
    if name in ['Support Vector Machine', 'Logistic Regression']:
        X_train_used = X_train_scaled
        X_test_used = X_test_scaled
    else:
        X_train_used = X_train
        X_test_used = X_test

    print(f"\nОбучение {name}...")
    model.fit(X_train_used, y_train)
    y_pred = model.predict(X_test_used)
    accuracy = accuracy_score(y_test, y_pred)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_model = model

    results.append({
        'Model': name,
        'Accuracy': accuracy
    })

    print(f"{name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Визуализация результатов
print("\n" + "=" * 50)
print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 50)

# Сравнение точности моделей
plt.figure(figsize=(10, 6))
models_names = [result['Model'] for result in results]
accuracies = [result['Accuracy'] for result in results]

bars = plt.bar(models_names, accuracies, color=['lightblue', 'lightgreen', 'lightcoral'])
plt.title('Сравнение точности моделей')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Добавляем значения на столбцы
for bar, accuracy in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{accuracy:.3f}', ha='center', va='bottom')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Матрица ошибок для лучшей модели
print(f"\nЛУЧШАЯ МОДЕЛЬ: {best_model_name} (Accuracy: {best_accuracy:.4f})")

if best_model_name in ['Support Vector Machine', 'Logistic Regression']:
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title(f'Матрица ошибок - {best_model_name}')
plt.ylabel('Истинные значения')
plt.xlabel('Предсказанные значения')
plt.show()

# Визуализация важности признаков (для Random Forest)
if best_model_name == 'Random Forest' and hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(10, 6))
    feature_importance = best_model.feature_importances_
    feature_names = X.columns
    indices = np.argsort(feature_importance)[::-1]

    plt.bar(range(len(feature_importance)), feature_importance[indices], color='lightseagreen')
    plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Важность признаков (Random Forest)')
    plt.ylabel('Важность')
    plt.tight_layout()
    plt.show()

# Матрица корреляций
plt.figure(figsize=(12, 8))
correlation_matrix = data_encoded.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Матрица корреляций')
plt.tight_layout()
plt.show()

# Распределения признаков
print("\nВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЙ ПРИЗНАКОВ")

# Выбираем только числовые колонки для гистограмм
numeric_columns = data_encoded.select_dtypes(include=[np.number]).columns

# Создаем subplot для более компактного отображения
n_cols = 3
n_rows = (len(numeric_columns) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
if n_rows == 1:
    axes = [axes] if n_cols == 1 else axes
else:
    axes = axes.flatten()

for i, column in enumerate(numeric_columns):
    if i < len(axes):
        sns.histplot(data=data_encoded, x=column, kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Распределение {column}')
        axes[i].tick_params(axis='x', rotation=45)

# Скрываем пустые subplot
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# Анализ целевой переменной
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
y_series = pd.Series(y)
y_series.value_counts().plot(kind='bar', color=['lightblue', 'lightcoral'])
plt.title('Распределение целевой переменной')
plt.xlabel('Diagnosis (0=No, 1=Yes)')
plt.ylabel('Количество')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
# Box plot для числовых признаков по целевой переменной
if len(numeric_columns) > 1:
    # Выбираем первый числовой признак (кроме целевой переменной)
    feature_candidates = [col for col in numeric_columns if col != 'Diagnosis']
    if feature_candidates:
        feature = feature_candidates[0]
        data_boxplot = data_encoded.copy()
        data_boxplot['Diagnosis'] = y
        sns.boxplot(data=data_boxplot, x='Diagnosis', y=feature)
        plt.title(f'{feature} по Diagnosis')

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("АНАЛИЗ ЗАВЕРШЕН!")
print(f"Лучшая модель: {best_model_name} с точностью {best_accuracy:.4f}")
print("=" * 50)