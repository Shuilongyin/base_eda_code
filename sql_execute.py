# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:56:58 2017

@author: memedai
"""

import pandas as pd
import pyodbc
import pymysql



class MysqlExecute():
    def __init__(self,host,user,passwd,db,port):
        self.host=host
        self.user=user
        self.passwd=passwd
        self.db=db
        self.port=port
    def db_select_table(self,sql):
        try:
            conn = pymysql.connect(host=self.host, user=self.user,
                                       passwd=self.passwd, db=self.db, port=self.port, charset='utf8')
            table = pd.read_sql_query(sql,conn)
            #frame = pd.read_sql_query(sql,conn)
        finally:
            conn.close()
        return table
    def db_select(self,sql):
        try:
            conn = pymysql.connect(host=self.host, user=self.user,
                                       passwd=self.passwd, db=self.db, port=self.port, charset='utf8')
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()
        finally:
            cursor.close()
            conn.close()
        return 
    def db_insert(self,sql,rows):
        try:
            conn = pymysql.connect(host=self.host, user=self.user,
                               passwd=self.passwd, db=self.db, port=self.port, charset='utf8')
            cursor = conn.cursor()
            cursor.executemany(sql,rows)
            conn.commit()
        finally:
            cursor.close()
            conn.close()
        return
#    def db_insert_df(self,df,table_name,if_exists,chunksize):
#        try:
#            engine = create_engine('mysql+mysqlconnector://[{}]:[{}]@[{}]:[{}]/[{}]'
#                                .format(self.user,self.passwd,self.host,self.port,self.db)
#                                , echo=False)
#            cnx = engine.raw_connection()
#            df.to_sql(name=table_name,con=cnx,if_exists=if_exists,index=False,chunksize=chunksize)
#        finally:
#            cnx.close()
#        return

class MssqlExecute():
    def __init__(self,driver,server,database,uid,pwd):
        self.driver = driver
        self.server = server
        self.database = database
        self.uid = uid
        self.pwd = pwd
    def db_insert(self,sql,rows):
        try:
            conn=pyodbc.connect('DRIVER={};SERVER={};DATABASE={};UID={};PWD={}'
                                  .format(self.driver,self.server,self.database,self.uid,self.pwd))
            cursor = conn.cursor()
            cursor.executemany(sql,rows)
            conn.commit()
        finally:
            cursor.close()
            conn.close()
        return
    def db_select(self,sql):
        try:
            conn=pyodbc.connect('DRIVER={};SERVER={};DATABASE={};UID={};PWD={}'
                                  .format(self.driver,self.server,self.database,self.uid,self.pwd))
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()
        finally:
            cursor.close()
            conn.close()
        return