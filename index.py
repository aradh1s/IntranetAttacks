
from flask import Flask, render_template, request,flash
import pandas as pd
from flask import session
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
import pickle
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as plt4
from sklearn.model_selection import train_test_split


from DBConfig import DBConnection

from LSTM import lstm_evaluation
from ANN import ann_evaluation
from CNN import cnn_evaluation
import numpy as np
from keras.models import load_model


app = Flask(__name__)
app.secret_key = "abc"


dict={}

accuracy_list=[]
accuracy_list.clear()
precision_list=[]
precision_list.clear()
recall_list=[]
recall_list.clear()
f1score_list=[]
f1score_list.clear()
permissions = []

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/admin")
def admin():
    return render_template("admin.html")


@app.route("/adminlogin_check",methods =["GET", "POST"])
def adminlogin():

        uid = request.form.get("unm")
        pwd = request.form.get("pwd")
        if uid=="admin" and pwd=="admin":

            return render_template("admin_home.html")
        else:
            return render_template("admin.html",msg="Invalid Credentials")




@app.route("/evaluations")
def evaluations():

    ann_list = []
    cnn_list = []
    lstm_list = []
    metrics=[]

    accuracy_ann, precision_ann, recall_ann, fscore_ann = ann_evaluation()
    ann_list.append("ANN")
    ann_list.append(accuracy_ann)
    ann_list.append(precision_ann)
    ann_list.append(recall_ann)
    ann_list.append(fscore_ann)

    accuracy_cnn, precision_cnn, recall_cnn, fscore_cnn = cnn_evaluation()
    cnn_list.append("CNN")
    cnn_list.append(accuracy_cnn)
    cnn_list.append(precision_cnn)
    cnn_list.append(recall_cnn)
    cnn_list.append(fscore_cnn)

    accuracy_lstm, precision_lstm, recall_lstm, fscore_lstm = lstm_evaluation()
    lstm_list.append("LSTM")
    lstm_list.append(accuracy_lstm)
    lstm_list.append(precision_lstm)
    lstm_list.append(recall_lstm)
    lstm_list.append(fscore_lstm)



    metrics.clear()

    metrics.append(ann_list)
    metrics.append(cnn_list)
    metrics.append(lstm_list)


    return render_template("evaluations.html", evaluations=metrics)





@app.route("/perevaluations")
def perevaluations():
    accuracy_graph()
    precision_graph()
    recall_graph()
    f1score_graph()
    return render_template("metrics.html")



def accuracy_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    accuracy_list.clear()

    cursor.execute("select accuracy from evaluations")
    acdata=cursor.fetchall()

    for record in acdata:
        accuracy_list.append(float(record[0]))

    height = accuracy_list

    bars = ('ANN','CNN','LSTM')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height, color=['red','green', 'blue'])
    plt.xticks(y_pos, bars)
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy')
    plt.title('Analysis of DL models Accuracies')
    plt.savefig('static/accuracy.png')
    plt.clf()


    return ""


def precision_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()

    cursor.execute("select precesion from evaluations")
    pdata = cursor.fetchall()

    precision_list.clear()
    for record in pdata:
        precision_list.append(float(record[0]))

    height = precision_list
    print("pheight=",height)
    bars = ('ANN', 'CNN', 'LSTM')
    y_pos = np.arange(len(bars))
    plt2.bar(y_pos, height, color=['green', 'brown', 'violet'])
    plt2.xticks(y_pos, bars)
    plt2.xlabel('Algorithms')
    plt2.ylabel('Precision')
    plt2.title('Analysis of DL models Precisions')
    plt2.savefig('static/precision.png')
    plt2.clf()
    return ""

def recall_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    recall_list.clear()
    cursor.execute("select recall from evaluations")
    recdata = cursor.fetchall()

    for record in recdata:
        recall_list.append(float(record[0]))

    height = recall_list

    bars = ('ANN', 'CNN', 'LSTM')
    y_pos = np.arange(len(bars))
    plt3.bar(y_pos, height, color=['orange', 'cyan', 'violet'])
    plt3.xticks(y_pos, bars)
    plt3.xlabel('Algorithms')
    plt3.ylabel('Recall')
    plt3.title('Analysis of DL models Recalls')
    plt3.savefig('static/recall.png')
    plt3.clf()
    return ""


def f1score_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    f1score_list.clear()

    cursor.execute("select f1score from evaluations")
    fsdata = cursor.fetchall()

    for record in fsdata:
        f1score_list.append(float(record[0]))

    height = f1score_list

    bars = ('ANN', 'CNN', 'LSTM')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height, color=['brown', 'green', 'orange'])
    plt.xticks(y_pos, bars)
    plt.xlabel('Algorithms')
    plt.ylabel('F1-Score')
    plt.title('Analysis of DL models F1-Score')
    plt4.savefig('static/f1score.png')
    plt4.clf()
    return ""


@app.route("/newuser")
def newuser():
    return render_template("register.html")

@app.route("/user")
def user():
    return render_template("user.html")


@app.route("/user_register",methods =["GET", "POST"])
def user_register():
    try:
        sts=""
        name = request.form.get('name')
        uid = request.form.get('unm')
        pwd = request.form.get('pwd')
        mno = request.form.get('mno')
        email = request.form.get('email')
        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from register where userid='" + uid + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:
            sts = 0
        else:
            sql = "insert into register values(%s,%s,%s,%s,%s)"
            values = (name,uid, pwd,email,mno)
            cursor.execute(sql, values)
            database.commit()
            sts = 1

        if sts==1:
            return render_template("user.html", msg="Registered Successfully..! Login Here.")


        else:
            return render_template("register.html", msg="User name already exists..!")



    except Exception as e:
        print(e)

    return ""

@app.route("/userlogin_check",methods =["GET", "POST"])
def userlogin_check():

        uid = request.form.get("unm")
        pwd = request.form.get("pwd")

        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from register where userid='" + uid + "' and passwrd='" + pwd + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:
            session['uid'] = uid

            return render_template("user_home.html")
        else:

            return render_template("user.html", msg2="Invalid Credentials")

        return ""


@app.route("/detection")
def detection():
    return render_template("prediction.html")

@app.route("/prediction", methods =["GET", "POST"])
def prediction():
    test_fname = request.form.get("file")

    print(test_fname)

    testframe = pd.read_csv(test_fname)

    del testframe['type']

    testdata = testframe
    print(testdata)

    model = open('lstm.h5', 'rb')
    
    lstm = pickle.load(model)

    result = lstm.predict(testdata)

    print("res=", result)

    int2label={0:'ddos' ,1:"dos",2:"injection",3:"mitm",4:"normal",5:"password",6:"scanning",7:"xss"}


    return render_template("prediction.html", result=int2label[result[0]])






if __name__ == '__main__':
    app.run(host="localhost", port=2244, debug=True)
