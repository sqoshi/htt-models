import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

with open(
    "/home/piotr/Workspaces/studies/hands-to-text/hands_to_text/data.pickle", "rb"
) as f:
    data_dict = pickle.load(f)

print(
    len(data_dict["data"]),
    len(data_dict["labels"]),
)
print([len(k) for k in data_dict["data"]])
data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)
model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print("{}% of samples were classified correctly !".format(score * 100))

with open("model.pickle", "wb") as f:
    pickle.dump({"model": model}, f)
