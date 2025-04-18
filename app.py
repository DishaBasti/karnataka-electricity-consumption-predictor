from flask import Flask, render_template, request
from model import linear_reg, knn_reg, rf_reg, ada_reg
import csv

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('cover.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/data')
def data():
    return render_template("data.html")

@app.route('/model')
def model():
    return render_template("model.html")

@app.route('/dataset')
def dataset():
    csv_data = []
    with open(r'D:\Shreya Files\Projects\ML\templates\energyKarnataka.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            csv_data.append(row)
    return render_template('dataset.html', csv_data=csv_data)

@app.route('/result', methods=['POST'])
def result():
    a = request.form['name']
    b = int(request.form['year'])
    c= request.form['district']
    d = int(request.form['plot_size'])
    e= int(request.form['household_size'])
    f = request.form['model']
    if f=='Linear_Regression':
        x, y, z, w = linear_reg(b,c,d,e)
        return render_template('result.html',a=a,b=b,c=c,d=d,e=e,f=f,x=x,y=y, z=z, w=w)
    elif f=="KNN_Regression":
        x, y, z, w = knn_reg(b,c,d,e)
        return render_template('result.html',a=a,b=b,c=c,d=d,e=e,f=f,x=x,y=y,z=z,w=w)
    elif f=="Random_Forest_Regression":
        x, y, z, w = rf_reg(b,c,d,e)
        return render_template('result.html',a=a,b=b,c=c,d=d,e=e,f=f,x=x,y=y,z=z,w=w)
    elif f=="AdaBoost_Regression":
        x, y, z, w = ada_reg(b,c,d,e)
        return render_template('result.html',a=a,b=b,c=c,d=d,e=e,f=f,x=x,y=y,z=z,w=w)
    #elif f=="XGBoost_Regression":
       # x, y, z, w = xgb_reg(b,c,d,e)
        #return render_template('result.html',a=a,b=b,c=c,d=d,e=e,f=f,x=x,y=y,z=z,w=w)

if __name__ == '__main__':
    app.run()
