# -*- coding: utf-8 -*-

import configparser
from utility import mysql_handler as mysql


USER_NAME = "real"
PASSWORD = "vmfl515!dnlf"
HOST_NAME = "sorinegi-cluster.cluster-ro-ce1us4oyptfa.ap-northeast-2.rds.amazonaws.com"  # 'mathflat-dev-cluster.cluster-ce1us4oyptfa.ap-northeast-2.rds.amazonaws.com'
PORT = "3306"
DB_NAME = "iclass"
TABLE_NAME = "Table_middle_book_data"  # Table_middle_book_data (시중교재) / Table_middle_problems (문제은행)


db = mysql.MysqlHandler(
    USER_NAME,
    PASSWORD,
    hostname=HOST_NAME,
    port=int(PORT),
    database=DB_NAME,
    logger=None,
    show_=True,
)

db_colum_names = db.select_column_names(TABLE_NAME)
print("DB column names : {}".format(db_colum_names))

cond_list = [
    "{0}<={1}".format("unitCode", "212072"),
]
filter_string = db.create_filter_string(cond_list=cond_list)
print(filter_string)
db_data = db.select_with_filter(TABLE_NAME, filter_string=filter_string)
print("DB data size : {}".format(len(db_data)))
