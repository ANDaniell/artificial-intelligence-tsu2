import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

# Загрузка данных с правильной кодировкой
try:
    df = pd.read_csv('kaggle_income.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('kaggle_income.csv', encoding='latin-1')

# Предобработка данных
# Преобразование числовых колонок
numeric_cols = ['Mean', 'Median', 'Stdev', 'sum_w', 'ALand', 'AWater', 'Lat', 'Lon']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Преобразование категориальных данных
categorical_cols = ['State_Name', 'State_ab', 'County', 'City', 'Zip_Code', 'Area_Code']
df[categorical_cols] = df[categorical_cols].astype('category')

# Анализ пропущенных значений
print("Пропущенные значения:")
print(df.isnull().sum().sort_values(ascending=False))

# Основные статистики
print("\nСтатистики по доходам:")
print(df[['Mean', 'Median', 'Stdev']].describe())

# Визуализация распределения доходов
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['Median'], bins=50, kde=True, color='blue')
plt.title('Распределение медианного дохода')
plt.xlabel('Доллары США')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['Median'])
plt.title('Ящик с усами для медианного дохода')
plt.tight_layout()
plt.show()

# Анализ по штатам
top_states = df.groupby('State_Name', observed=False)['Median'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_states.values, y=top_states.index, palette='viridis', legend=False)

plt.title('Топ-10 штатов по медианному доходу')
plt.xlabel('Средний медианный доход ($)')
plt.ylabel('Штат')
plt.show()

# Корреляционный анализ
corr_matrix = df[['Mean', 'Median', 'Stdev', 'sum_w', 'ALand', 'AWater']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Корреляция между показателями')
plt.show()

# Анализ городских показателей
city_analysis = df.groupby('City').agg({
    'Median': 'mean',
    'sum_w': 'sum',
    'Lat': 'first',
    'Lon': 'first'
}).sort_values('Median', ascending=False).head(10)

print("\nТоп-10 городов по медианному доходу:")
print(city_analysis)
