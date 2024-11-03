from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from numpy import unique
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import os
#from DBConfig import DBConnection

def cnn_evaluation():
    '''db = DBConnection.getConnection()
    cursor = db.cursor()'''
    x, y = training_features()
    x = np.array(x, dtype=np.float16)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20,random_state=27)

    model = Sequential()
    model.add(Conv1D(32, 2, activation="relu", input_shape=(124, 1)))
    model.add(Dense(16, activation="relu"))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer="adam",
                  metrics=['accuracy'])

    xtrain = xtrain + np.random.normal(0, 2.2, size=xtrain.shape)
    #model.summary()
    model.fit(xtrain, ytrain, batch_size=32, epochs=10, verbose=1)
    '''pred = model.predict(xtest)
    pred_y = pred.argmax(axis=-1)

    accuracy = accuracy_score(ytest, pred_y) * 100
    precision = precision_score(ytest, pred_y,average="macro") * 100
    recall = recall_score(ytest, pred_y,average="macro") * 100
    fscore = f1_score(ytest, pred_y,average="macro") * 100'''
    #ytest = np.argmax(ytest, axis=1)
    y_pred = np.argmax(model.predict(xtest), axis=1)

    acc = accuracy_score(ytest, y_pred) * 100

    precsn = precision_score(ytest, y_pred, average="macro") * 100

    recall = recall_score(ytest, y_pred, average="macro") * 100

    f1score = f1_score(ytest, y_pred, average="macro") * 100
    print("CNN=",acc,precsn,recall,f1score)
    #values = ("CNN", str(accuracy), str(precision), str(recall), str(fscore))
    '''sql = "insert into evaluations values(%s,%s,%s,%s,%s)"
    cursor.execute(sql, values)
    db.commit()'''


    print("[INFO] Saving model...")

    model.save("cnn_model.h5")

    #return acc, precision, recall, fscore



def training_features():
    training_df = pd.read_csv("../IntranetAttacks/dataset.csv", sep=',')

    data = training_df.drop('label', axis=1)

    # Replace empty strings or spaces with NaN
    data.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # Drop rows with any missing values
    data = data.dropna()

    x_train = data.iloc[:, :-1]

    y_train = data.iloc[:, -1]

    scaler = StandardScaler()
    x_train= scaler.fit_transform(x_train)

    label_encoder = LabelEncoder()
    y_train= label_encoder.fit_transform(y_train)

    #y_train = one_hot_encoding(y_train.ravel())

    '''y_train = df["class"]

    del df["class"]

    x_train = df'''

    return x_train,y_train


def one_hot_encoding(data):
  L_E = LabelEncoder()
  integer_encoded = L_E.fit_transform(data)
  onehot_encoder = OneHotEncoder(sparse=False)
  integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
  one_hot_encoded_data = onehot_encoder.fit_transform(integer_encoded)
  return one_hot_encoded_data


cnn_evaluation()