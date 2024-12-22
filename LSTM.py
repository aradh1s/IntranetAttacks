import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
import os
from keras.models import load_model
from DBConfig import DBConnection
def training_features():
    # Step 1: Load the dataset
    data = pd.read_csv("../IntranetAttacks/dataset.csv", sep=',')

    # Step 2: Preprocess the dataset
    # Assuming the last column is the target label and others are features
    # Handle missing values (if any)
    # data.fillna(data.mean(), inplace=True)

    data = data.drop('label', axis=1)

    # Replace empty strings or spaces with NaN
    data.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # Drop rows with any missing values
    data = data.dropna()

    data.to_csv("datapreproc.csv", index=False)

    # Separate features and labels
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values  # Target labels

    # Encode labels (if categorical)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reshape data for LSTM (samples, timesteps, features)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    return X, y



def lstm_evaluation():

    db = DBConnection.getConnection()

    cursor = db.cursor()

    X, y = training_features()

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    if os.path.exists("../IntranetAttacks/lstm_model.h5"):
        model_path = 'lstm_model.h5'

        lstm_model = load_model(model_path)

        y_pred = np.argmax(lstm_model.predict(X_test), axis=1)

        acc = accuracy_score(y_test, y_pred) * 100

        precsn = precision_score(y_test, y_pred, average="macro") * 100

        recall = recall_score(y_test, y_pred, average="macro") * 100

        f1score = f1_score(y_test, y_pred, average="macro") * 100

        print("lstm=", acc, precsn, recall, f1score)

        values = ("LSTM", str(acc), str(precsn), str(recall), str(f1score))
        sql = "insert into evaluations values(%s,%s,%s,%s,%s)"
        cursor.execute(sql, values)
        db.commit()

    else:
        # Step 4: Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(8, activation='softmax'))  # Assuming binary classification

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        X_train = X_train + np.random.normal(0, 2.0, size=X_train.shape)

        # Step 5: Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32)

        model.save("lstm_model.h5")

        print("[INFO] Saving model...")

        y_pred = np.argmax(model.predict(X_test), axis=1)

        acc = accuracy_score(y_test, y_pred) * 100

        precsn = precision_score(y_test, y_pred, average="macro") * 100

        recall = recall_score(y_test, y_pred, average="macro") * 100

        f1score = f1_score(y_test, y_pred, average="macro") * 100



        print("lstm=", acc, precsn, recall, f1score)
        values = ("LSTM", str(acc), str(precsn), str(recall), str(f1score))
        sql = "insert into evaluations values(%s,%s,%s,%s,%s)"
        cursor.execute(sql, values)
        db.commit()

    return acc, precsn, recall, f1score



#lstm_evaluation()
