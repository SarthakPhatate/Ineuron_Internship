import pandas as pd
import datetime
import shutil
import numpy as np

def bulkPredict(file):
    with open('Logs/bulkPredectionLogs.txt','a') as f:
        f.write('-----------------------------------------------------------------------------------------------------\n')
        if file.endswith('.csv'):
            f.write(str(datetime.datetime.now()) + ' File name is correct\n')
            try:
                f.write(str(datetime.datetime.now()) + ' Reading File\n')
                df = pd.read_csv('bulk_pred/' + file)
                f.write(str(datetime.datetime.now()) + ' Reading Columns\n')
                if (df.shape[1] == 20 and df.shape[0] != 0):
                    try:
                        df.columns = ['status', 'duration', 'credit_history', 'purpose', 'amount', 'savings',
                                      'employment_duration',
                                      'installment_rate', 'personal_status_sex', 'other_debtors', 'present_residence',
                                      'property', 'age',
                                      'other_installment_plans', 'housing', 'number_credits', 'job', 'people_liable',
                                      'telephone', 'foreign_worker']
                        f.write(str(datetime.datetime.now()) + ' Columns Renaming Successful\n')
                    except Exception as e:
                        f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
                        shutil.move("bulk_pred/" + file, "BadPredictionFile/" + file)
                        return "Problem is colums reading please check file again and uplaod",False
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
                    except Exception as e:
                        f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
                        return "Problem is column nmaes please check file again and uplaod", False

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
                        # droping the duration feature
                        # df.drop(columns='duration', inplace=True)
                        # Arranging the Columns
                        data = df[num_col[:]]
                        # one-hot-encoding on categorical features
                        for i in categorical_col:
                            dum_col = pd.get_dummies(df[i], drop_first=True)
                            data = pd.concat([data, dum_col], axis=1)
                        f.write(str(datetime.datetime.now()) + ' One Hot Encoding Successfully\n')
                        return data,True
                    except Exception as e:
                        f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
                        return "Problem is preparing data please check file again and uplaod", False
                else:
                    f.write(str(datetime.datetime.now()) + ' Invalid Colums/Row Length {}\n'.format(df.shape[1]))
                    return "Problem is column length please check file again and uplaod", False
            except Exception as e:
                f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
                return "Problem is reading file please check file again and uplaod", False

        else:
            f.write(str(datetime.datetime.now()) + ' File name is incorrect\n')
            shutil.move("bulk_pred/" + file, "BadPredictionFile/" + file)
            return "Problem is filename please check file again and uplaod", False