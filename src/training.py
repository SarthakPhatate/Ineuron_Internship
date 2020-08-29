import datetime
import sqlite3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def train():
    with open('Logs/TrainingLogs.txt', 'a') as f:
        f.write('-------------------------------------------------------------------------------------------------------\n')
        try:
            con = sqlite3.connect('Data.db')
            f.write(str(datetime.datetime.now()) + ' Connection is established: Database connected successsfylly\n')
            f.write(str(datetime.datetime.now()) + ' Lodaing the data from database to dataframe\n')
            query = "SELECT * FROM data"
            df = pd.read_sql(query, con)
            # split the data input input and output
            f.write(str(datetime.datetime.now()) + ' Preparing feature column and target column\n')
            X = df.drop(columns='c55')
            Y = df['c55']

            f.write(str(datetime.datetime.now()) + ' Applying the SMOTE function to handle imbalace data\n')
            sm = SMOTE(random_state=2)
            X_train_res, y_train_res = sm.fit_sample(X.values, Y)

            f.write(str(datetime.datetime.now()) + ' Splitting The data into train data and test data\n')
            # Splitting the data into training and testing data
            x_train, x_test, y_train, y_test = train_test_split(X_train_res, y_train_res, test_size=0.28, random_state=355, shuffle=True)

            return x_train, x_test, y_train, y_test
        except Exception as e:
            f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
            return False

def log_reg(x_train, x_test, y_train, y_test):
    with open('Logs/TrainingLogs.txt', 'a') as f:
        try:
            # instantiate a logistic regression model, and fit with X and y
            log = LogisticRegression(max_iter=400)
            log.fit(x_train, y_train)
            # check the accuracy metrics
            acc = log.score(x_test,y_test)
            f.write(str(datetime.datetime.now()) + ' Logistic Regression model trained [accuracy: {}]\n'.format(acc))
            return log,str(acc * 100)[:4]
        except Exception as e:
            f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
            return False

def dec_tree(x_train, x_test, y_train, y_test):
    with open('Logs/TrainingLogs.txt', 'a') as f:
        try:
            # Traning the model without doing any pre processing
            clf = DecisionTreeClassifier()
            clf.fit(x_train, y_train)
            # check the accuracy metrics
            acc = clf.score(x_test, y_test)
            f.write(str(datetime.datetime.now()) + ' Decision Tree model trained [accuracy: {}]\n'.format(acc))
            return clf,str(acc * 100)[:4]
        except Exception as e:
            f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
            return False

def ran_for(x_train, x_test, y_train, y_test):
    with open('Logs/TrainingLogs.txt', 'a') as f:
        try:
            # initialising the random forest classifier
            rand_clf = RandomForestClassifier()
            rand_clf.fit(x_train, y_train)
            # check the accuracy metrics
            acc = rand_clf.score(x_test, y_test)
            f.write(str(datetime.datetime.now()) + ' Random Forest model trained [accuracy: {}]\n'.format(acc))
            return rand_clf,str(acc * 100)[:4]
        except Exception as e:
            f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
            return False

def xbg_class(x_train, x_test, y_train, y_test):
    with open('Logs/TrainingLogs.txt', 'a') as f:
        try:
            # fit model no training data
            xgb = XGBClassifier(objective='binary:logistic')
            xgb.fit(x_train, y_train)
            # check the accuracy metrics
            acc = xgb.score(x_test, y_test)
            f.write(str(datetime.datetime.now()) + ' XgBoost model trained [accuracy: {}]\n'.format(acc))
            return xgb,str(acc * 100)[:4]
        except Exception as e:
            f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
            return False