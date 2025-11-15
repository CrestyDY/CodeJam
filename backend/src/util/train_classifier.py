import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'data.pickle')
MODEL_PATH = os.path.join(SCRIPT_DIR, "models")

data_dict1 = pickle.load(open(os.path.join(DATA_PATH, 'one_hand_data.pickle'), 'rb'))
data_dict2 = pickle.load(open(os.path.join(DATA_PATH, 'two_hands.pickle'), 'rb'))

data1 = np.asarray(data_dict1['data'])
labels1 = np.asarray(data_dict1['labels'])
data2= np.asarray(data_dict2['data'])
labels2 = np.asarray(data_dict2['labels'])

x_train1, x_test1, y_train1, y_test1 = train_test_split(data1, labels1, test_size=0.2, shuffle=True, stratify=labels1)
x_train2, x_test2, y_train2, y_test2 = train_test_split(data2, labels2, test_size=0.2, shuffle=True, stratify=labels2)
model1 = RandomForestClassifier()
model2 = RandomForestClassifier()

model1.fit(x_train1, y_train1)
model2.fit(x_train2, y_train2)

y_predict1 = model1.predict(x_test1)
y_predict2 = model2.predict(x_test2)

score1 = accuracy_score(y_predict1, y_test1)
score2 = accuracy_score(y_predict2, y_test2)
print('{}% of one-handed samples were classified correctly !'.format(score1 * 100))

print('{}% of two-handed samples were classified correctly !'.format(score2 * 100))
f1 = open(os.path.join(MODEL_PATH + 'model_one_hand.p'), 'wb')
f2 = open(os.path.join(MODEL_PATH + 'model_two_hands.p'), 'wb')
pickle.dump({'model': model1}, f1)
pickle.dump({'model': model2}, f2)
f1.close()
f2.close()
