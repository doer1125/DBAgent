import json
import logging
import os
import sys
from typing import Annotated, TypedDict, List, Optional
from typing_extensions import NotRequired

import matplotlib.pyplot as plt
import networkx as nx
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.drawing.nx_agraph import graphviz_layout
import pickle
from DBConfig import DBConfig
from dotenv import load_dotenv

import mysql.connector
from mysql.connector import Error

# Configure the main logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Adjust logging levels for specific libraries to reduce noise
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

load_dotenv()


class Config:
    def __init__(self):
        # Load required environment variables
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

        # 初始化数据库配置
        self.db_config = DBConfig()

        # Ensure all required variables are set
        if not self.deepseek_api_key:
            raise ValueError("Missing required environment variable: DEEPSEEK_API_KEY")

        # 测试数据库连接
        if not self.db_config.test_connection():
            raise ValueError("MySQL database connection failed")

        # Configure database connection using SQLAlchemy engine
        # 使用SQLDatabase包装SQLAlchemy引擎
        self.db_engine = SQLDatabase(self.db_config.get_sqlalchemy_engine())

        # Set up language models with specific configurations
        # self.llm = ChatOpenAI(temperature=0)  # Default model (e.g., GPT-3.5)
        # self.llm_deepseek = ChatOpenAI(
        #     model="deepseek-chat",  # 使用 "model" 而不是 "model_name"
        #     base_url="https://api.deepseek.com/v1",
        #     api_key=self.deepseek_api_key,
        #     temperature=0
        # )

        # 先创建 OpenAI 客户端
        client = OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=self.deepseek_api_key
        )

        # 然后传递给 ChatOpenAI
        self.llm_deepseek = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            client=client  # 直接传入客户端
        )

    def get_db_connection(self):
        """获取原始MySQL连接（用于直接操作）"""
        return mysql.connector.connect(
            host=self.db_config.host,
            port=self.db_config.port,
            user=self.db_config.user,
            password=self.db_config.password,
            database=self.db_config.database
        )


if __name__ == "__main__":
    try:
        config = Config()
        print("配置初始化成功")
        print(f"数据库引擎: {config.db_engine}")
        # print(f"LLM模型: {config.llm.model_name}")
        print(f"DeepSeek LLM模型: {config.llm_deepseek.model}")
    except Exception as e:
        print(f"配置初始化失败: {e}")
