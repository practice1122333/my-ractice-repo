import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Завантаження даних з інтернету
# Наприклад, дані з CSV-файлу на певному ресурсі (замініть на актуальне джерело)
url = "https://example.com/cucumber_prices.csv"  # Замініть на реальне джерело
try:
    data = pd.read_csv(url)
    print("Дані успішно завантажено!")
except Exception as e:
    print(f"Не вдалося завантажити дані: {e}")
    # Приклад вручну створених даних
    data = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'price': [20, 22, 23, 21, 25, 24, 27, 29, 28, 30]
    })

# Перетворення дат у числовий формат для регресії
data['days'] = (data['date'] - data['date'].min()).dt.days

# Розділення на X (фактори) і y (цільова змінна)
X = data[['days']]
y = data['price']

# Розділення даних на тренувальні та тестові набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Лінійна регресія
model = LinearRegression()
model.fit(X_train, y_train)

# Прогноз
y_pred = model.predict(X_test)

# Оцінка якості моделі
mse = mean_squared_error(y_test, y_pred)
print(f"Середньоквадратична похибка: {mse:.2f}")

# Побудова графіка
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['days'], y=data['price'], label="Фактичні дані")
plt.plot(data['days'], model.predict(data[['days']]), color='red', label="Лінійна регресія")
plt.xlabel("Кількість днів від початку")
plt.ylabel("Ціна огірків (грн/кг)")
plt.title("Лінійна регресія: ціна огірків на ринку")
plt.legend()
plt.grid()
plt.show()
