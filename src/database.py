import sqlite3
import datetime
from src import columns

def sql_connection(df):
    with open('Logs/PreprocessingLogs.txt', 'a') as f:
        try:
            con = sqlite3.connect('Data.db')
            f.write(str(datetime.datetime.now()) + " Connection is established: Database connected successsfylly\n")
            cursorObj = con.cursor()
            cursorObj.execute("CREATE TABLE if not exists data ({})".format(columns.TABLE_ATTRIBUTES))
            con.commit()
            df.columns = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21', 'c22', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c30', 'c31', 'c32', 'c33', 'c34', 'c35', 'c36', 'c37', 'c38', 'c39', 'c40', 'c41', 'c42', 'c43', 'c44', 'c45', 'c46', 'c47', 'c48', 'c49', 'c50', 'c51', 'c52', 'c53', 'c54','c55']
            df.to_sql('data', con, if_exists='append', index=False)
            f.write(str(datetime.datetime.now()) + ' Data Insertion in database successful\n')
            con.close()
            return True
        except Exception as e:
            f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
            return False