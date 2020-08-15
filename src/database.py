import sqlite3
import datetime

def sql_connection(df):
    with open('Training_Logs/TrainingLogs.txt', 'a') as f:
        try:
            con = sqlite3.connect('Data.db')
            f.write(str(datetime.datetime.now()) + " Connection is established: Database connected successsfylly\n")
            cursorObj = con.cursor()
            cursorObj.execute("CREATE TABLE if not exists data(c1 DOUBLE,c2 DOUBLE,c3 FLOAT,c4 FLOAT,c5 FLOAT,c6 FLOAT,c7 FLOAT,c8 FLOAT,c9 FLOAT,"
                              "c10 FLOAT,c11 FLOAT,c12 FLOAT,c13 FLOAT,c14 FLOAT,c15 FLOAT,c16 FLOAT,c17 FLOAT,c18 FLOAT,c19 FLOAT,"
                              "c20 FLOAT,c21 FLOAT,c22 FLOAT,c23 FLOAT,c24 FLOAT,c25 FLOAT,c26 FLOAT,c27 FLOAT,c28 FLOAT,c29 FLOAT,"
                              "c30 FLOAT,c31 FLOAT,c32 FLOAT,c33 FLOAT,c34 FLOAT,c35 FLOAT,c36 FLOAT,c37 FLOAT,c38 FLOAT,c39 FLOAT,"
                              "c40 FLOAT,c41 FLOAT,c42 FLOAT,c43 FLOAT,c44 FLOAT,c45 FLOAT,c46 FLOAT,c47 FLOAT,c48 FLOAT,c49 FLOAT,"
                              "c50 FLOAT,c51 FLOAT,c52 FLOAT,c53 FLOAT,c54 FLOAT)")
            con.commit()
            df.columns = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21', 'c22', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c30', 'c31', 'c32', 'c33', 'c34', 'c35', 'c36', 'c37', 'c38', 'c39', 'c40', 'c41', 'c42', 'c43', 'c44', 'c45', 'c46', 'c47', 'c48', 'c49', 'c50', 'c51', 'c52', 'c53', 'c54']
            df.to_sql('data', con, if_exists='append', index=False)
            f.write(str(datetime.datetime.now()) + ' Data Insertion in database successful\n')
            con.close()
            return True
        except Exception as e:
            f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
            return False