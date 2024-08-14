import pickle
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from helpers import TFIDFExtrator, preprocess_text


# Topics on dataset: "Hardware", "HR Support", "Access", "Miscellaneous", "Storage", "Purchase", "Internal Project", "Administrative rights"

dataset = pd.read_csv("dataset/all_tickets_processed_improved_v3.csv")
dataset_test = pd.read_csv("dataset/test.csv")

print("Dataset shape", dataset.shape)


dataset["Document"] = dataset["Document"].apply(preprocess_text)
dataset_test["Document"] = dataset_test["Document"].apply(preprocess_text)

print(dataset["Topic_group"].value_counts())

x_train = dataset["Document"]
y_train = dataset["Topic_group"]


clf = Pipeline(
    [
        ("features", TFIDFExtrator(column="Document", max_features=256)),
        ("clf", RandomForestClassifier(n_estimators=512, max_depth=64, n_jobs=-1)),
    ]
)

clf.fit(x_train, y_train)


x_test = dataset_test["Document"]
y_test = dataset_test["Topic_group"]

y_pred = clf.predict(x_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


with open("src/machine_learning_training/model.pkl", "wb") as f:
    pickle.dump(clf, f)
