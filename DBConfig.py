import os
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

class DBConfig:
    def __init__(self):
        self.host = os.getenv("MYSQL_HOST")
        self.port = int(os.getenv("MYSQL_PORT"))
        self.user = os.getenv("MYSQL_USER")
        self.password = os.getenv("MYSQL_PASSWORD")
        self.database = os.getenv("MYSQL_DATABASE")

        # 验证必要的环境变量
        if not all([self.user, self.password, self.database]):
            raise ValueError("Missing required MySQL environment variables: MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE")

    def get_connection_string(self):
        """返回SQLAlchemy连接字符串"""
        return f"mysql+mysqlconnector://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def get_sqlalchemy_engine(self):
        """返回SQLAlchemy引擎"""
        return create_engine(self.get_connection_string())

    def test_connection(self):
        """测试数据库连接"""
        try:
            conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tb_user")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            return True
        except Error as e:
            print(f"MySQL连接测试失败: {e}")
            return False


if __name__ == "__main__":
    config = DBConfig()
    config.test_connection()
