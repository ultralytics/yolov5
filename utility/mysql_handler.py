#!/usr/bin/env python
# -*- coding: utf-8 -*-
import mysql.connector
from utility import general_utils as g_utils
import pandas as pd
import sys
import time
import traceback

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 3306


class MysqlHandler:
    def __init__(
        self,
        username,
        passwd,
        hostname=DEFAULT_HOST,
        port=DEFAULT_PORT,
        database=None,
        logger=None,
        show_=False,
    ):

        self.connection = None
        self.hostname = hostname
        self.port = port
        self.username = username
        self.passwd = passwd
        self.db_name = database

        self.logger = logger
        if self.logger is None:
            self.logger = g_utils.setup_logger(None, None, logger_=False)

        self.connect_db()

    def connect_db(self):
        try:
            self.logger.info(
                " # try connect db : {}:{}. db:{}".format(
                    self.hostname, self.port, self.db_name
                )
            )
            self.connection = mysql.connector.connect(
                host=self.hostname,
                port=self.port,
                user=self.username,
                passwd=self.passwd,
                database=self.db_name,
            )
            self.logger.info(" # completed to connect db")
            return True
        except Exception as e:
            self.log_exception("connect_db", e)
            return False

    def close(self):
        try:
            if self.connection.is_connected():
                self.connection.close()
        except Exception as e:
            self.log_exception("close db connection", e)

    def get_cursor(self):
        for idx in range(3):
            try:
                cursor = self.connection.cursor()
                return cursor
            except Exception as e:
                self.log_exception("get_cursor", e)
                self.logger.info(
                    "get_cursor. try gain to connect db. {}".format(idx + 1)
                )

                self.close()

                time.sleep(1)
                for con_idx in range(3):
                    if self.connect_db():
                        break
                    self.logger.error(
                        "get_cursor. failed connection. try again. {}".format(con_idx)
                    )
                    time.sleep(1)

        self.logger.error("get_cursor. finally failed.")
        return None

    def create_database(self, db_name):
        try:
            cursor = self.get_cursor()
            if cursor is None:
                self.logger.error("create_database. failed to get cursor")
                return

            cursor.execute("CREATE DATABASE " + db_name)
            cursor.close()
        except Exception as e:
            self.log_exception("create_database", e)

    def show_databases(self):
        try:
            cursor = self.get_cursor()
            if cursor is None:
                self.logger.error("show_databases. failed to get cursor")
                return

            cursor.execute("SHOW DATABASES")
            self.print_cursor(cursor)
            cursor.close()
        except Exception as e:
            self.log_exception("show_databases", e)

    def use_database(self, db_name, show_=False):
        if show_:
            self.logger.info(' # Use "{}" database.'.format(db_name))
        try:
            cursor = self.get_cursor()
            if cursor is None:
                self.logger.error("use_database. failed to get cursor")
                return

            cursor.execute("USE " + db_name)
            cursor.close()
        except Exception as e:
            self.log_exception("use_database", e)

    def create_table(
        self,
        table_name,
        csv_fname,
        skip_row_num=0,
        skip_col_num=0,
        auto_inc_pk_=True,
        show_=False,
    ):
        if show_:
            self.logger.info(' # Create "{}" table.'.format(table_name))

        g_utils.file_exists(csv_fname, exit_=True)
        df = pd.read_csv(csv_fname, skiprows=skip_row_num)
        df = df.drop(df.columns[[x for x in range(skip_col_num)]], axis=1)
        sql = "CREATE TABLE " + table_name + " ("
        if auto_inc_pk_:
            sql += "id INT AUTO_INCREMENT PRIMARY KEY,"

        sql += " {} {}".format(df.loc[0][0].strip(), df.loc[0][1].strip())
        if isinstance(df.loc[0][2], str) and len(df.loc[0][2]) > 0:
            sql += " " + df.loc[0][2].strip()

        for row in range(1, df.shape[0]):
            if not isinstance(df.loc[row][0], str):
                break

            name = df.loc[row][0].strip()
            type = df.loc[row][1].strip()

            if name == "PRIMARY KEY":
                sql += ", CONSTRAINT {}_pk PRIMARY KEY".format(table_name)
                sql += "(" + type + ")"
            else:
                sql += ", {}".format(name)
                sql += " {}".format(type)
                if (
                    isinstance(df.loc[row][2], str) and len(df.loc[row][2]) > 0
                ):  # additional info
                    sql += " " + df.loc[row][2].strip()

        sql += " )"

        self.logger.info("create table. sql:{}".format(sql))

        try:
            cursor = self.get_cursor()
            if cursor is None:
                self.logger.error("create_table. failed to get cursor")
                return False

            cursor.execute(sql)
            cursor.close()
            return True
        except Exception as e:
            self.logger.error("create_table.failed : {}".format(e))
            return False

    def describe_table(self, table_name):
        cursor = self.get_cursor()
        if cursor is None:
            self.logger.error("describe_table. failed to get cursor")
            return

        cursor.execute("DESCRIBE " + table_name)
        self.logger.info("")
        self.logger.info(" # TABLE SCHEMA :{}".format(table_name))
        self.print_cursor(cursor)
        cursor.close()

    def show_table(self):
        cursor = self.get_cursor()
        if cursor is None:
            self.logger.error("show_table. failed to get cursor")
            return

        cursor.execute("SHOW TABLES")
        self.print_cursor(cursor)
        cursor.close()

    def drop_table(self, table_name, show_=False):

        if show_:
            self.logger.info(' # Drop "{}" table.'.format(table_name))

        try:
            cursor = self.get_cursor()
            if cursor is None:
                self.logger.error("drop_table. failed to get cursor")
                return

            cursor.execute("DROP TABLE " + " IF EXISTS " + table_name)
            cursor.close()
        except Exception as e:
            self.log_exception("drop_table", e)
            sys.exit(1)

    @staticmethod
    def get_string_with_quotes(values):
        values_str = ""
        for value in values:
            values_str += "'" + value + "', "
        return values_str[:-2]

    def insert_into_table(self, table_name, value_dicts, logging=False):
        if len(value_dicts) == 0:
            return False

        sql = "INSERT INTO " + table_name + " (" + ", ".join(value_dicts.keys()) + ") "
        sql += "VALUES (" + self.get_string_with_quotes(value_dicts.values()) + ")"

        try:
            cursor = self.get_cursor()
            if cursor is None:
                self.logger.error("insert_into_table. failed to get cursor")
                return False

            cursor.execute(sql)
            self.connection.commit()
            self.print_cursor(cursor)
            cursor.close()
            if logging:
                self.logger.info("")
                self.logger.info(" # insert_into_table : {}".format(sql))
        except Exception as e:
            self.log_exception("insert_into_table", e)

        return True

    def insert_csv_file_into_table(self, table_name, csv_fname, row_num=-1):

        g_utils.file_exists(csv_fname, exit_=True)
        df = pd.read_csv(csv_fname, dtype=object)

        sql = "INSERT INTO " + table_name + " (" + ", ".join(df.columns) + ") "
        sql += "VALUES (" + ", ".join(["%s"] * len(df.columns)) + ")"

        if row_num < 0:
            row_num = df.shape[0]

        vals = []
        for i in range(row_num):
            if not isinstance(df.loc[i][0], str):
                break
            val = []
            for j in range(len(df.columns)):
                col = df.loc[i][j]
                if not isinstance(col, str):
                    col = ""
                val.append(col)
            vals.append(tuple(val))

        try:
            cursor = self.get_cursor()
            if cursor is None:
                self.logger.error("insert_csv_file_into_table. failed to get cursor")
                return False

            cursor.executemany(sql, vals)
            self.connection.commit()
            self.print_cursor(cursor)
            self.logger.info("")
            self.logger.info(
                " # insert_csv_file_into_table: {:d} was inserted.".format(
                    cursor.rowcount
                )
            )
            cursor.close()
        except Exception as e:
            self.log_exception("insert_csv_file_into_table", e)

        return True

    def update_column_into_row(
        self, table_name, col_name, col_cond, row_name, row_cond, show_=False
    ):

        if show_:
            self.logger.info(' # update_column_into_row "{}" table.'.format(table_name))

        try:
            cursor = self.get_cursor()
            if cursor is None:
                self.logger.error("update_column_into_row. failed to get cursor")
                return False

            sql = (
                "UPDATE "
                + table_name
                + " SET "
                + col_name
                + " = '"
                + col_cond
                + "'"
                + " WHERE "
                + row_name
                + " = '"
                + row_cond
                + "'"
            )
            cursor.execute(sql)
            self.connection.commit()
            self.logger.info(
                " # Update column into row : {:d} record(s) affected.".format(
                    cursor.rowcount
                )
            )
            cursor.close()
        except Exception as e:
            self.log_exception("update_column_into_row", e)

        return True

    def update_columns_into_row(
        self, table_name, value_dicts, filter_string, show_=False
    ):

        if show_:
            self.logger.info(' # Update "{}" table.'.format(table_name))

        try:
            cursor = self.get_cursor()
            if cursor is None:
                self.logger.error("update_columns_into_row. failed to get cursor")
                return False

            sql = "UPDATE " + table_name + " SET "

            value_list = []
            for col_name in value_dicts.keys():
                value_list.append(col_name + " = '" + value_dicts[col_name] + "'")

            sql += ", ".join(value_list)
            sql += " WHERE " + filter_string

            cursor.execute(sql)
            self.connection.commit()
            self.logger.info(
                " # Update columns into row : table:{}. {:d} record(s) affected.".format(
                    table_name, cursor.rowcount
                )
            )
            cursor.close()
        except Exception as e:
            self.log_exception("update_columns_into_row", e)
            return False

        return True

    def select_all(self, table_name, show_=False, inplace=False):

        try:
            cursor = self.get_cursor()
            if cursor is None:
                self.logger.error("select_all. failed to get cursor")
                return

            cursor.execute("SELECT * FROM " + table_name)
            result = cursor.fetchall()
            cursor.close()
            self.logger.info("")
            self.logger.info(" # Select ALL")
            if show_:
                self.print_cursor(result)
            if inplace is False:
                return result
        except Exception as e:
            self.log_exception("select_all", e)

    def execute(self, query, show_=False):

        try:
            cursor = self.get_cursor()
            if cursor is None:
                self.logger.error("execute. failed to get cursor")
                return []

            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            if show_:
                self.print_cursor(result)
            return result
        except Exception as e:
            self.log_exception("execute", e)
            return []

    def select_columns(self, table_name, column_names, show_=False, inplace=False):

        try:
            cursor = self.get_cursor()
            if cursor is None:
                self.logger.error("select_columns. failed to get cursor")
                return []

            cursor.execute("SELECT " + ", ".join(column_names) + " FROM " + table_name)
            result = cursor.fetchall()
            cursor.close()
            if show_:
                self.print_cursor(result)
            if inplace is False:
                return result
        except Exception as e:
            self.log_exception("select_columns", e)
            return []

    def select_column_names(self, table_name, show_=False, inplace=False):

        try:
            cursor = self.get_cursor()
            if cursor is None:
                self.logger.error("select_column names. failed to get cursor")
                return []

            sql = (
                "SELECT column_name FROM information_schema.columns WHERE TABLE_NAME = '"
                + table_name
                + "'"
            )
            cursor.execute(sql)
            result = cursor.fetchall()
            cursor.close()
            if show_:
                self.print_cursor(result)
            if inplace is False:
                return_values = []
                for item in result:
                    return_values.append(item[0])
                return return_values
        except Exception as e:
            self.log_exception("select_columns", e)
            return []

    def create_filter_string(self, cond_list, sort_col="", sort_desc=True):
        filter_string = ""
        if cond_list:
            for cond in cond_list:
                filter_string += " AND " + cond
        if sort_col:
            filter_string += " ORDER BY " + sort_col
            if sort_desc:
                filter_string += " desc "
            else:
                filter_string += " asc "
        return filter_string

    def select_with_filter(
        self, table_name, filter_string, col_names=None, show_=False, inplace=False
    ):

        try:
            sql = "SELECT " + ", ".join(col_names) if col_names else "SELECT *"
            sql += " FROM " + table_name + " WHERE 1=1 " + filter_string

            cursor = self.get_cursor()
            if cursor is None:
                self.logger.error("select_with_filter. failed to get cursor")
                return []

            cursor.execute(sql)
            result = cursor.fetchall()
            cursor.close()
            if show_:
                self.print_cursor(result)
            if inplace is False:
                return result
        except Exception as e:
            self.log_exception("select_with_filter", e)
            return []

    def delete_with_filter(self, table_name, filter_string, show_=False):
        try:
            sql = "DELETE FROM " + table_name + " WHERE " + filter_string

            if show_:
                self.logger.info(sql)
            cursor = self.get_cursor()
            if cursor is None:
                self.logger.error("delete_with_filter. failed to get cursor")
                return

            cursor.execute(sql)
            cursor.close()
        except Exception as e:
            self.log_exception("delete_with_filter", e)

    def print_cursor(self, result):
        for x in result:
            self.logger.info(x)

    def log_exception(self, msg, e):
        self.logger.error(
            msg + ". Exception: {}".format(e) + "\n" + traceback.format_exc()
        )
