import tkinter as tk
from tkinter import messagebox
from DiabetesClassifier import load_and_predict

class DiabetesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Диагностика диабета — Ввод данных")
        self.root.geometry("420x600")
        self.root.resizable(False, False)

        # Переменные для всех полей
        self.fields = {
            "Age": tk.DoubleVar(),
            "BMI": tk.DoubleVar(),
            "Chol": tk.DoubleVar(),
            "TG": tk.DoubleVar(),
            "HDL": tk.DoubleVar(),
            "LDL": tk.DoubleVar(),
            "Cr": tk.DoubleVar(),
            "BUN": tk.DoubleVar(),
        }

        # Переменная для пола (RadioButton)
        self.gender_var = tk.StringVar(value="M")

        title = tk.Label(root, text="Введите данные пациента", font=("Arial", 14, "bold"))
        title.pack(pady=10)

        form_frame = tk.Frame(root)
        form_frame.pack(padx=10, pady=5)

        # Добавляем все numeric поля
        row = 0
        for label, var in self.fields.items():
            tk.Label(form_frame, text=f"{label}:", font=("Arial", 11)).grid(row=row, column=0, sticky="e", pady=5)
            entry = tk.Entry(form_frame, textvariable=var, width=20, font=("Arial", 10))
            entry.grid(row=row, column=1, padx=5, pady=5)
            row += 1

        # Радиокнопки для Gender
        tk.Label(form_frame, text="Gender:", font=("Arial", 11)).grid(row=row, column=0, sticky="e", pady=5)
        gender_frame = tk.Frame(form_frame)
        gender_frame.grid(row=row, column=1, pady=5)

        tk.Radiobutton(gender_frame, text="Мужской", variable=self.gender_var, value="M",
                       font=("Arial", 10)).pack(side="left", padx=5)
        tk.Radiobutton(gender_frame, text="Женский", variable=self.gender_var, value="F",
                       font=("Arial", 10)).pack(side="left", padx=5)

        self.predict_button = tk.Button(
            root, text="Проверить", font=("Arial", 12, "bold"),
            bg="#4CAF50", fg="white", width=20, command=self.make_prediction
        )
        self.predict_button.pack(pady=20)

    def make_prediction(self):
        try:
            patient_data = {}

            # Добавляем числовые значения
            for key, var in self.fields.items():
                value = var.get()
                patient_data[key] = float(value)

            # Добавляем выбранный пол
            patient_data["Gender"] = self.gender_var.get()

            result = load_and_predict(patient_data)

            ResultWindow(self.root, result)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при вводе данных:\n{e}")

class ResultWindow:
    def __init__(self, master, result):
        self.window = tk.Toplevel(master)
        self.window.title("Результат")
        self.window.geometry("400x400")
        self.window.resizable(False, False)

        tk.Label(self.window, text="Результаты нейросети", font=("Arial", 14, "bold")).pack(pady=10)

        frame = tk.Frame(self.window)
        frame.pack(pady=10)

        diagnosis = result['diagnosis']
        confidence = result['confidence']
        prob0 = result['probability_class_0']
        prob1 = result['probability_class_1']
        risk = result['risk_level']

        tk.Label(frame, text=f"Диагноз: {diagnosis}", font=("Arial", 12, "bold")).pack(pady=5)
        tk.Label(frame, text=f"Уверенность: {confidence:.1%}", font=("Arial", 11)).pack(pady=5)
        tk.Label(frame, text=f"Вероятность диабета: {prob1:.1%}", font=("Arial", 11)).pack(pady=5)
        tk.Label(frame, text=f"Вероятность отсутствия диабета: {prob0:.1%}", font=("Arial", 11)).pack(pady=5)

        color = {"Низкий риск": "green", "Средний риск": "orange", "Высокий риск": "red"}[risk]
        tk.Label(frame, text=f"Уровень риска: {risk}",
                 font=("Arial", 12, "bold"), fg=color).pack(pady=10)

        tk.Button(self.window, text="Закрыть", width=15, command=self.window.destroy,
                  bg="#607D8B", fg="white").pack(pady=15)


if __name__ == "__main__":
    root = tk.Tk()
    app = DiabetesApp(root)
    root.mainloop()
