import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
warnings.filterwarnings('ignore')

class NeuralDiabetesClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        self.imputer = SimpleImputer(strategy='median')
        self.encoder_model = None
        self.history = None

    def load_and_prepare_data(self, data_path):
        print(f"Загружаем данные из {data_path}")
        data = pd.read_csv(data_path)
        if 'Diagnosis' not in data.columns:
            raise ValueError("CSV файл должен содержать колонку 'Diagnosis'")
        print(f"Размер данных: {data.shape}")
        print(f"Колонки: {list(data.columns)}")
        print(f"Пропущенные значения:\n{data.isnull().sum()}")
        return data

    def build_autoencoder(self, input_dim, encoding_dim=8):
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        encoded = tf.keras.layers.Dense(32, activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_layer)
        encoded = tf.keras.layers.BatchNormalization()(encoded)
        encoded = tf.keras.layers.Dropout(0.2)(encoded)
        encoded = tf.keras.layers.Dense(16, activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(encoded)
        encoded = tf.keras.layers.BatchNormalization()(encoded)
        encoded = tf.keras.layers.Dropout(0.2)(encoded)
        encoded = tf.keras.layers.Dense(encoding_dim, activation='relu', name='bottleneck')(encoded)
        decoded = tf.keras.layers.Dense(16, activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(encoded)
        decoded = tf.keras.layers.BatchNormalization()(decoded)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        decoded = tf.keras.layers.Dense(32, activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(decoded)
        decoded = tf.keras.layers.BatchNormalization()(decoded)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        decoded = tf.keras.layers.Dense(input_dim, activation='linear')(decoded)
        autoencoder = tf.keras.Model(input_layer, decoded)
        encoder = tf.keras.Model(input_layer, encoded)
        return autoencoder, encoder

    def build_classifier(self, input_dim):
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        x = tf.keras.layers.Dense(64, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(32, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(16, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        classifier = tf.keras.Model(input_layer, output)
        return classifier

    def advanced_feature_engineering(self, data):
        data_eng = data.copy()
        if 'Age' in data_eng.columns and 'BMI' in data_eng.columns:
            data_eng['Age_BMI_Interaction'] = data_eng['Age'] * data_eng['BMI']
        if 'Glucose' in data_eng.columns and 'BMI' in data_eng.columns:
            data_eng['Glucose_BMI_Ratio'] = data_eng['Glucose'] / data_eng['BMI']
        if 'HDL' in data_eng.columns and 'LDL' in data_eng.columns:
            data_eng['Chol_Ratio'] = data_eng['HDL'] / data_eng['LDL']
        if 'Age' in data_eng.columns:
            data_eng['Age_Group'] = pd.cut(data_eng['Age'],
                                           bins=[0, 30, 45, 60, 100],
                                           labels=[0, 1, 2, 3])
        return data_eng

    def clean_data(self, data):
        data_clean = data.copy()
        data_clean = data_clean.replace([np.inf, -np.inf], np.nan)
        numeric_columns = data_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'Diagnosis':
                if data_clean[col].isnull().any():
                    data_clean[col].fillna(data_clean[col].median(), inplace=True)
        data_clean = data_clean.dropna(subset=['Diagnosis'])
        return data_clean

    def prepare_data(self, data):
        data_encoded = data.copy()
        categorical_columns = data_encoded.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'Diagnosis':
                data_encoded[col] = self.label_encoder.fit_transform(data_encoded[col].astype(str))
        return data_encoded

    def train_autoencoder(self, X_train, epochs=100):
        print("Обучение автоэнкодера...")
        input_dim = X_train.shape[1]
        autoencoder, encoder = self.build_autoencoder(input_dim)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss='mse',
                            metrics=['mae'])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
        ]
        history_ae = autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        print("Автоэнкодер обучен!")
        return encoder

    def train(self, data_path, epochs=100, use_autoencoder=True):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow не установлен. Установите: pip install tensorflow")
        data = self.load_and_prepare_data(data_path)
        data_clean = self.clean_data(data)
        data_eng = self.advanced_feature_engineering(data_clean)
        data_processed = self.prepare_data(data_eng)
        X = data_processed.drop('Diagnosis', axis=1)
        y = data_processed['Diagnosis']
        self.feature_names = X.columns.tolist()
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        if use_autoencoder:
            self.encoder_model = self.train_autoencoder(X_train_scaled, epochs=50)
            X_train_encoded = self.encoder_model.predict(X_train_scaled)
            X_test_encoded = self.encoder_model.predict(X_test_scaled)
            print(f"Размерность после энкодера: {X_train_encoded.shape}")
            X_train_final = X_train_encoded
            X_test_final = X_test_encoded
            input_dim = X_train_encoded.shape[1]
        else:
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled
            input_dim = X_train_scaled.shape[1]
        print("Обучение классификатора...")
        self.model = self.build_classifier(input_dim)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_accuracy'),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, monitor='val_loss')
        ]
        self.history = self.model.fit(
            X_train_final, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test_final, y_test),
            callbacks=callbacks,
            verbose=1
        )
        y_pred_proba = self.model.predict(X_test_final)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"\nНейросетевая модель обучена успешно!")
        print(f"Точность на тестовой выборке: {accuracy:.4f}")
        print(f"AUC-ROC: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        self.plot_training_history()
        self.plot_confusion_matrix(y_test, y_pred)
        self.is_trained = True
        return accuracy, auc_score

    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'])
        plt.title('Confusion Matrix - Neural Network')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

    def predict(self, input_data):
        if not self.is_trained:
            raise Exception("Модель не обучена! Сначала вызовите метод train().")
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data.copy()
        else:
            raise ValueError("Input data должен быть dict или DataFrame")
        input_eng = self.advanced_feature_engineering(input_df)
        for col in input_eng.select_dtypes(include=['object']).columns:
            if col in input_eng.columns and col != 'Diagnosis':
                try:
                    input_eng[col] = self.label_encoder.transform(input_eng[col].astype(str))
                except ValueError:
                    input_eng[col] = 0
        for feature in self.feature_names:
            if feature not in input_eng.columns:
                input_eng[feature] = 0
        input_eng = input_eng[self.feature_names]
        input_eng = input_eng.fillna(0)
        input_scaled = self.scaler.transform(input_eng)
        if self.encoder_model is not None:
            input_final = self.encoder_model.predict(input_scaled)
        else:
            input_final = input_scaled
        probability = self.model.predict(input_final)[0][0]
        prediction = 1 if probability > 0.5 else 0
        confidence = probability if prediction == 1 else (1 - probability)
        return {
            'prediction': int(prediction),
            'confidence': float(confidence),
            'probability_class_0': float(1 - probability),
            'probability_class_1': float(probability),
            'diagnosis': 'Диабет' if prediction == 1 else 'Нет диабета',
            'risk_level': self.get_risk_level(probability)
        }

    def get_risk_level(self, probability):
        if probability < 0.3:
            return "Низкий риск"
        elif probability < 0.7:
            return "Средний риск"
        else:
            return "Высокий риск"

    def save_model(self, filepath='neural_diabetes_model'):
        if not self.is_trained:
            raise Exception("Нет обученной модели для сохранения!")
        os.makedirs(filepath, exist_ok=True)
        self.model.save(os.path.join(filepath, 'classifier.h5'))
        if self.encoder_model:
            self.encoder_model.save(os.path.join(filepath, 'encoder.h5'))
        model_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, os.path.join(filepath, 'preprocessor.joblib'))
        print(f"Модель сохранена в {filepath}")

    def load_model(self, filepath='neural_diabetes_model'):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Папка модели {filepath} не найдена!")
        self.model = tf.keras.models.load_model(os.path.join(filepath, 'classifier.h5'))
        encoder_path = os.path.join(filepath, 'encoder.h5')
        if os.path.exists(encoder_path):
            self.encoder_model = tf.keras.models.load_model(encoder_path)
        model_data = joblib.load(os.path.join(filepath, 'preprocessor.joblib'))
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        print(f"Модель загружена из {filepath}")

def main():
    classifier = NeuralDiabetesClassifier()
    try:
        accuracy, auc = classifier.train('Diabetes Classification.csv',
                                         epochs=100,
                                         use_autoencoder=True)
        classifier.save_model('neural_diabetes_model')
        print(f"\nОбучение завершено! Модель сохранена.")
        print(f"Точность: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
    except FileNotFoundError:
        print("Файл 'Diabetes Classification.csv' не найден!")
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")

def load_and_predict(input_data, model_path='neural_diabetes_model'):
    classifier = NeuralDiabetesClassifier()
    classifier.load_model(model_path)
    return classifier.predict(input_data)

if __name__ == "__main__":
    main()