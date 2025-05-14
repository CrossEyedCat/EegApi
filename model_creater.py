import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Загрузка данных из CSV
# Ожидается, что CSV-файл содержит колонки: EEG, cum_time, d1, t1, t2, a1, a2, b1, b2, b3, g1, phase
data = pd.read_csv('eeg_data.csv')
print("Данные загружены. Колонки:", data.columns.tolist())

# 2. Выбор признаков и меток
# Будем использовать столбцы: EEG, d1, t1, t2, a1, a2, b1, b2, b3, g1 (исключаем cum_time)
feature_columns = ['EEG', 'd1', 't1', 't2', 'a1', 'a2', 'b1', 'b2', 'b3', 'g1']
X = data[feature_columns]
y = data['phase']

# 3. Кодирование меток (например: 'none' -> 0, 'left' -> 1, 'right' -> 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Кодировка меток:", dict(zip(le.classes_, le.transform(le.classes_))))

# 4. Нормализация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Разделение данных на обучающую и тестовую выборки (85%/15%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
)
print("Размер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)

# 6. Обучение модели (RandomForestClassifier)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Модель обучена.")

# 7. Предсказание на тестовой выборке и оценка качества модели
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

print("\nТочность модели на тестовых данных: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n", report)

# 8. Сохранение модели, скейлера и энкодера для дальнейшего использования
joblib.dump(clf, 'trained_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Обученная модель и объекты сохранены в файлы: 'trained_model.pkl', 'scaler.pkl', 'label_encoder.pkl'")
