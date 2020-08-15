import datetime
import pandas as pd
import joblib
import numpy as np
from src.database import sql_connection
from src import training

def train_validation(file):
    info = '[ info ] '
    yield info + "File Validation in Process, Please wait<br/><br/>\n"
    with open('Training_Logs/TrainingLogs.txt','a') as f:
        f.write("-------------------------------------------------------------------------------------------------------------")
        if file.endswith('.asc'):
            f.write(str(datetime.datetime.now()) + ' File name is correct\n')
            try:
                f.write(str(datetime.datetime.now()) + ' Reading File\n')
                df = pd.read_csv(file,sep=' ')
                f.write(str(datetime.datetime.now()) + ' Reading Columns\n')
                if(df.shape[1] == 21 and df.shape[0]!=0):
                    yield info + " File Validation Successfull<br/><br/>\n"
                    yield info + " Data Preprocessing Started, Please wait....<br/><br/>\n"
                    try:
                        df.columns = ['status', 'duration', 'credit_history', 'purpose', 'amount', 'savings',
                                      'employment_duration',
                                      'installment_rate', 'personal_status_sex', 'other_debtors', 'present_residence',
                                      'property', 'age',
                                      'other_installment_plans', 'housing', 'number_credits', 'job', 'people_liable',
                                      'telephone', 'foreign_worker',
                                      'credit_risk']
                        f.write(str(datetime.datetime.now()) + ' Columns Renaming Successful\n')
                        yield info + "Column Name Changed successful<br/><br/>\n"
                    except Exception as e:
                        f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
                        yield info + "Problem in renaming Columns, Please Check log file and retry uploading<br/><br/>\n"

                    try:
                        # assigning the appropriate categories labels to the data of each feature
                        df['status'].replace(to_replace=[1, 2, 3, 4],
                                             value=["no checking account", "..<0 DM", "0<=..<200 DM",
                                                    "..>= 200 DM/salary for at least 1 year"],
                                             inplace=True)

                        df['credit_history'].replace(to_replace=[0, 1, 2, 3, 4],
                                                     value=["delay in paying off in the past",
                                                            "critical account/other credits elsewhere",
                                                            "no credits taken/all credits paid back duly",
                                                            "existing credits paid back duly till now",
                                                            "all credits at this bank paid back duly"], inplace=True)

                        df['purpose'].replace(to_replace=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                              value=["others", "car (new)", "car (used)", "furniture/equipment",
                                                     "radio/television",
                                                     "domestic appliances", "repairs", "education", "vacation",
                                                     "retraining", "business"], inplace=True)

                        df['savings'].replace(to_replace=[1, 2, 3, 4, 5],
                                              value=["unknown/no savings account", "..<100 DM", "100<=..<500 DM",
                                                     "500<=..<1000 DM", "..>=1000 DM"], inplace=True)

                        df['other_debtors'].replace(to_replace=[1, 2, 3], value=["None", "co-applicant", "guarantor"],
                                                    inplace=True)

                        df['personal_status_sex'].replace(to_replace=[1, 2, 3, 4],
                                                          value=["male : divorced/separated",
                                                                 "female : non-single or male : single",
                                                                 "male : married/widowed", "female : single"],
                                                          inplace=True)

                        df['installment_rate'].replace(to_replace=[1, 2, 3, 4],
                                                       value=[">=35", "25<=..<35", "20<=..<25", "<20"], inplace=True)

                        df['present_residence'].replace(to_replace=[1, 2, 3, 4],
                                                        value=["<1 yr", "1<=..<4 yrs", "4<=..<7 yrs", ">=7 yrs"],
                                                        inplace=True)

                        df['property'].replace(to_replace=[1, 2, 3, 4],
                                               value=["unknown / no property", "car or other",
                                                      "building soc. savings agr./life insurance",
                                                      "real estate"], inplace=True)

                        df['other_installment_plans'].replace(to_replace=[1, 2, 3], value=["bank", "stores", "none"],
                                                              inplace=True)

                        df['housing'].replace(to_replace=[1, 2, 3], value=["for free", "rent", "own"], inplace=True)

                        df['number_credits'].replace(to_replace=[1, 2, 3, 4], value=["1", "2-3", "4-5", ">= 6"],
                                                     inplace=True)

                        df['job'].replace(to_replace=[1, 2, 3, 4],
                                          value=["unemployed/unskilled - non-resident", "unskilled - resident",
                                                 "skilled employee/official",
                                                 "manager/self-empl./highly qualif. employee"], inplace=True)

                        df['employment_duration'].replace(to_replace=[1, 2, 3, 4, 5],
                                                          value=["unemployed", "<1 yr", "1<=..<4 yrs", "4<=..<7 yrs",
                                                                 ">=7 yrs"], inplace=True)

                        df['people_liable'].replace(to_replace=[1, 2], value=["3 or more", "0 to 2"], inplace=True)

                        df['telephone'].replace(to_replace=[1, 2], value=["No", "Yes"], inplace=True)

                        df['foreign_worker'].replace(to_replace=[1, 2], value=["yes", "no"], inplace=True)

                        f.write(str(datetime.datetime.now()) + ' data labeling Successful\n')
                        yield info + "data Labeling Successful<br/><br/>\n"
                    except Exception as e:
                        f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
                        yield info + "Problem in labeling features, Please Check log file and retry uploading<br/><br/>\n"

                    try:
                        # defining the categorial columns
                        categorical_col = ['status', 'credit_history', 'purpose', 'savings', 'installment_rate',
                                           'employment_duration',
                                           'personal_status_sex', 'other_debtors', 'present_residence', 'property',
                                           'other_installment_plans',
                                           'housing', 'number_credits', 'job', 'people_liable', 'telephone',
                                           'foreign_worker']
                        # defining the num,erica columns
                        num_col = ['duration', 'amount', 'age']
                        # fixing the Skewness and outliers using log function
                        for col in num_col:
                            df[col] = np.log1p(df[col])
                        f.write(str(datetime.datetime.now()) + ' Skewness removed Successfully\n')
                        yield info + "Skewness removed Successful<br/><br/>\n"
                        # droping the duration feature
                        df.drop(columns='duration', inplace=True)
                        # Arranging the Columns
                        data = df[num_col[1:]]
                        # one-hot-encoding on categorical features
                        for i in categorical_col:
                            dum_col = pd.get_dummies(df[i], drop_first=True)
                            data = pd.concat([data, dum_col], axis=1)
                        data = pd.concat([data, df['credit_risk']], axis=1)
                        f.write(str(datetime.datetime.now()) + ' One Hot Encoding Successfully\n')
                        yield info + "One Hot Encoding Successfully<br/><br/>\n"
                    except Exception as e:
                        f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
                        yield info + "Problem in feature Selection, Please Check log file and retry uploading<br/><br/>\n"

                    try:
                        yield info + "Saving Preprocessed File<br/><br/>\n"
                        data.to_csv('data_preprocessed/prossed.csv', index=False)
                        f.write(str(datetime.datetime.now()) + ' File Saved Successfully\n')
                        if sql_connection(data):
                            yield info + "Saved into Database<br/><br/>\n"
                        else:
                            yield info + "Error occured while inserting into Database. Please check log file<br/><br/>\n"
                    except Exception as e:
                        f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
                        yield info + "Problem in saving Preprocessed Data, Please Check log file and retry uploading<br/><br/>\n"
                else:
                    f.write(str(datetime.datetime.now()) + ' Invalid Colums/Row Length {}\n'.format(df.shape[1]))
            except Exception as e:
                f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))

        else:
            f.write(str(datetime.datetime.now()) + ' File name is incorrect\n')
            yield info + " File name is incorrect, Please upload correct file<br/><br/>\n"

        try:
            f.write(str(datetime.datetime.now()) + ' Training the model\n')
            yield info + " Training started Please wait!<br/><br/>\n"
            x_train, x_test, y_train, y_test = training.train()

            f.write(str(datetime.datetime.now()) + ' Training Logistic Regression model\n')
            yield info + " Training on Logistic Regression Model<br/><br/>\n"
            log_model, acc1 = training.log_reg(x_train, x_test, y_train, y_test)
            yield info + "accuracy = {}<br/><br/>\n".format(acc1)

            f.write(str(datetime.datetime.now()) + ' Training Decision Tree model\n')
            yield info + " Training on Decision Tree Model<br/><br/>\n"
            dec_model, acc2 = training.dec_tree(x_train, x_test, y_train, y_test)
            yield info + "accuracy = {}<br/><br/>\n".format(acc2)

            f.write(str(datetime.datetime.now()) + ' Training Random Forest model\n')
            yield info + " Training on Random Forest Model<br/><br/>\n"
            ran_model, acc3 = training.ran_for(x_train, x_test, y_train, y_test)
            yield info + "accuracy = {}<br/><br/>\n".format(acc3)

            f.write(str(datetime.datetime.now()) + ' Training XGboost Classifier model\n')
            yield info + " Training on XGboost Classifier Model<br/><br/>\n"
            xgb_model, acc4 = training.xbg_class(x_train, x_test, y_train, y_test)
            yield info + "accuracy = {}<br/><br/>\n".format(acc4)

            yield info + 'Training Completed<br/><br/>\n'
            dict_ = {acc1:log_model, acc2:dec_model, acc3:ran_model, acc4:xgb_model}
            joblib.dump(dict_[max(dict_)], 'model/model.sav')
            yield info + "Saved best performing model"
        except Exception as e:
            f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
            yield info + "Problem in Training, Please Check log file and retry<br/><br/>\n"