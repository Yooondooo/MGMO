# simple_predict.py - упрощенный скрипт для предсказаний
from DiabetesClassifier import load_and_predict

#30,F,22,5.4,1.7,1.4,3.3,53,5.7,0
patient_data = {
    'Age': 30,
    'Gender': 'F',
    'BMI': 22,
    'Chol': 5.4,
    'TG': 1.7,
    'HDL': 1.4,
    'LDL': 3.3,
    'Cr': 53,
    'BUN': 5.7
}
# 31,M,37,4.1,2.2,0.7,2.4,60,3,1
# patient_data = {
#     'Age': 31,
#     'Gender': 'M',
#     'BMI': 37,
#     'Chol': 4.1,
#     'TG': 2.2,
#     'HDL': 0.7,
#     'LDL': 2.4,
#     'Cr': 60,
#     'BUN': 3.1
# }
# 52,F,33,8.5,0.8,6.6,1.3,88,6.4,1

#Факторы риска
#BMI > 25, Chol 5.5-6.5 и выше, TG > 1.8, Cr > 90, BUN > 5.0
# patient_data = {
#     'Age': 52,
#     'Gender': 'F',
#     'BMI': 33,
#     'Chol': 8.5,
#     'TG': 0.8,
#     'HDL': 6.6,
#     'LDL': 1.3,
#     'Cr': 88,
#     'BUN': 6.4
# }
# Получаем предсказание
result = load_and_predict(patient_data)

# Вывод результата
print(f"Диагноз: {result['diagnosis']}")
print(f"Уверенность: {result['confidence']:.1%}")
print(f"Вероятность диабета: {result['probability_class_1']:.1%}")