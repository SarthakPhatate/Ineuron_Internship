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
    with open('Training_Logs/TrainingLogs.txt', 'a') as f:
        try:
            con = sqlite3.connect('Data.db')
            f.write(str(datetime.datetime.now()) + " Connection is established: Database connected successsfylly\n")
            f.write(str(datetime.datetime.now()) + ' Lodaing the data from database to dataframe\n')
            query = "SELECT * FROM data"
            df = pd.read_sql(query, con)
            # split the data input input and output
            f.write(str(datetime.datetime.now()) + ' Preparing feature column and target column\n')
            X = df.drop(columns='c54')
            Y = df['c54']

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
    with open('Training_Logs/TrainingLogs.txt', 'a') as f:
        # instantiate a logistic regression model, and fit with X and y
        log = LogisticRegression(max_iter=400)
        log.fit(x_train, y_train)
        acc = log.score(x_test,y_test)
        # check the accuracy metrics
        return log,str(acc * 100)[:4]

def dec_tree(x_train, x_test, y_train, y_test):
    with open('Training_Logs/TrainingLogs.txt', 'a') as f:
        # Traning the model without doing any pre processing
        clf = DecisionTreeClassifier()
        clf.fit(x_train, y_train)
        acc = clf.score(x_test, y_test)
        return clf,str(acc * 100)[:4]

def ran_for(x_train, x_test, y_train, y_test):
    with open('Training_Logs/TrainingLogs.txt', 'a') as f:
        # initialising the random forest classifier
        rand_clf = RandomForestClassifier()
        rand_clf.fit(x_train, y_train)
        acc = rand_clf.score(x_test, y_test)
        # check the accuracy metrics
        return rand_clf,str(acc * 100)[:4]

def xbg_class(x_train, x_test, y_train, y_test):
    with open('Training_Logs/TrainingLogs.txt', 'a') as f:
        # fit model no training data
        xgb = XGBClassifier(objective='binary:logistic')
        xgb.fit(x_train, y_train)
        acc = xgb.score(x_test, y_test)
        # check the accuracy metrics
        return xgb,str(acc * 100)[:4]