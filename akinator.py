import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class AkinatorModel:
    def __init__(self, db_path):
        self.df = self.load_data(db_path)
        self.X, self.y, self.le_name, self.label_encoders = self.prepare_data()
        self.clf = self.train_classifier()
        self.asked_questions = set()

    def load_data(self, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT * FROM people")
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        df = pd.DataFrame(rows, columns=column_names)
        conn.close()
        return df

    def prepare_data(self):
        le_name = LabelEncoder().fit(self.df["name"])
        label_encoders = {
            col: LabelEncoder().fit(pd.concat([self.df[col], pd.Series(['unknown'])]))
            for col in self.df.columns if
            col != "name"}

        self.df["name"] = le_name.transform(self.df["name"])
        for col in label_encoders:
            self.df[col] = label_encoders[col].transform(self.df[col])

        x = self.df.drop("name", axis=1)
        y = self.df["name"]

        return x, y, le_name, label_encoders

    def train_classifier(self):
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(self.X, self.y)
        return clf

    def get_top_n_names_and_probabilities(self, predicted_probs, n=5):
        top_n_indices = predicted_probs.argsort()[0][-n:][::-1]
        names = self.le_name.inverse_transform(top_n_indices)
        probabilities = predicted_probs[0][top_n_indices]
        return list(zip(names, probabilities))

    def custom_transform(self, label_encoder, data):
        transformed_data = []
        for item in data:
            if item in label_encoder.classes_:
                transformed_data.append(label_encoder.transform([item])[0])
            else:
                transformed_data.append(-1)
        return transformed_data

    def predict_person(self, input_data):
        new_data = pd.DataFrame([input_data])

        for col in self.label_encoders:
            new_data[col] = self.custom_transform(self.label_encoders[col], [new_data[col].iloc[0]])

        predicted_probs = self.clf.predict_proba(new_data)
        return predicted_probs

    def get_most_important_question(self):
        feature_importances = self.clf.feature_importances_
        max_importance_index = feature_importances.argmax()
        most_important_question = self.X.columns[max_importance_index]
        return most_important_question

    def get_top_n_important_questions(self, n):
        feature_importances = self.clf.feature_importances_
        top_n_indices = feature_importances.argsort()[-n:][::-1]
        top_n_questions = self.X.columns[top_n_indices]
        return top_n_questions

    def select_best_question(self, input_data):
        initial_probs = self.predict_person(input_data)[0]
        initial_entropy = self.entropy(initial_probs)

        best_question = None
        max_entropy_reduction = -np.inf

        answered_questions = [question for question, answer in input_data.items() if answer is not None]

        for question in self.label_encoders.keys():
            if question not in answered_questions and question not in self.asked_questions:
                temp_input_data = input_data.copy()

                entropy_reduction = 0
                for answer in ['yes', 'no', 'unknown']:
                    temp_input_data[question] = answer
                    new_probs = self.predict_person(temp_input_data)[0]
                    new_entropy = self.entropy(new_probs)
                    entropy_reduction += initial_entropy - new_entropy

                if entropy_reduction > max_entropy_reduction:
                    max_entropy_reduction = entropy_reduction
                    best_question = question

        if best_question is not None:
            self.asked_questions.add(best_question)

        return best_question

    def entropy(self, probabilities):
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))
