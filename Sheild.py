import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the ClassLabel.csv file
class_labels_df = pd.read_csv("ClassLabel.csv")

# Load .asm and .bytes files
asm_files = []
bytes_files = []
for filename in os.listdir("."):
    if filename.endswith(".asm"):
        with open(filename, "r", errors="ignore") as asm_file:
            asm_files.append(asm_file.read())
    elif filename.endswith(".bytes"):
        with open(filename, "r", errors="ignore") as bytes_file:
            bytes_files.append(bytes_file.read())

# Combine .asm and .bytes files into a single feature
combined_features = [asm + " " + bytes for asm, bytes in zip(asm_files, bytes_files)]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(combined_features)

# Merge features with class labels
X_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
X_df["Class"] = class_labels_df["Class"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_df.drop(columns=["Class"]), X_df["Class"], test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
