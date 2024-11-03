# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
#from DBConfig import DBConnection
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def ann_evaluation():
    #db = DBConnection.getConnection()
    #cursor = db.cursor()
    #cursor.execute("delete from evaluations")
    # load the dataset
    x, y = training_features()
    x = np.array(x, dtype=np.float16)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)

    #ytrain = one_hot_encoding(ytrain.ravel())
    #ytest = one_hot_encoding(ytest.ravel())

    # define the keras model
    model = Sequential()
    model.add(Dense(32, input_shape=(124,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    # compile the keras model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    xtrain = xtrain + np.random.normal(2, 1.5, size=xtrain.shape)
    # fit the keras model on the dataset
    model.fit(xtrain, ytrain, epochs=20, batch_size=16)

    #ytest = np.argmax(ytest)
    y_pred = np.argmax(model.predict(xtest), axis=1)

    acc = accuracy_score(ytest, y_pred) * 100

    precsn = precision_score(ytest, y_pred, average="macro") * 100

    recall = recall_score(ytest, y_pred, average="macro") * 100

    f1score = f1_score(ytest, y_pred, average="macro") * 100

    print(acc)
    print(precsn)
    print(recall)
    print(f1score)

    print("ANN=", acc, precsn, recall, f1score)
    values = ("ANN", str(acc), str(precsn), str(recall), str(f1score))
    sql = "insert into evaluations values(%s,%s,%s,%s,%s)"
    cursor.execute(sql, values)
    db.commit()

    return acc,precsn,recall,f1score


def training_features():

    training_df = pd.read_csv("../IntranetAttacks/dataset.csv", sep=',')


    data=training_df.drop('label',axis=1)


    #Replace empty strings or spaces with NaN
    data.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # Drop rows with any missing values
    data=data.dropna()

    x_train = data.iloc[:, :-1]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)


    y_train = data.iloc[:, -1]

    return x_train,y_train



ann_evaluation()