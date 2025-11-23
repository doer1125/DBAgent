import json
import logging
import os
import re
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
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.drawing.nx_agraph import graphviz_layout
import pickle
from DBConfig import DBConfig
from dotenv import load_dotenv
import httpx
import openai

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


# 彻底修补所有客户端
def patch_all_clients():
    """修补同步和异步客户端"""

    # 修补同步客户端
    original_sync_init = httpx.Client.__init__

    def patched_sync_init(self, *args, **kwargs):
        kwargs.pop('proxies', None)
        kwargs.pop('proxy', None)
        return original_sync_init(self, *args, **kwargs)

    httpx.Client.__init__ = patched_sync_init

    # 修补异步客户端 - 这是关键！
    original_async_init = httpx.AsyncClient.__init__

    def patched_async_init(self, *args, **kwargs):
        kwargs.pop('proxies', None)
        kwargs.pop('proxy', None)
        return original_async_init(self, *args, **kwargs)

    httpx.AsyncClient.__init__ = patched_async_init

    # 修补 OpenAI 的内部客户端
    if hasattr(openai, '_base_client'):
        # 修补同步客户端
        original_openai_sync = openai._base_client.SyncAPIClient.__init__

        def patched_openai_sync(self, **kwargs):
            kwargs.pop('proxies', None)
            return original_openai_sync(self, **kwargs)

        openai._base_client.SyncAPIClient.__init__ = patched_openai_sync

        # 修补异步客户端
        original_openai_async = openai._base_client.AsyncAPIClient.__init__

        def patched_openai_async(self, **kwargs):
            kwargs.pop('proxies', None)
            return original_openai_async(self, **kwargs)

        openai._base_client.AsyncAPIClient.__init__ = patched_openai_async


class Config:
    def __init__(self):
        # Load required environment variables
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        # 初始化数据库配置
        self.db_config = DBConfig()

        self.db = self.db_config.database

        # Ensure all required variables are set
        if not self.deepseek_api_key:
            raise ValueError("Missing required environment variable: DEEPSEEK_API_KEY")

        # 测试数据库连接
        if not self.db_config.test_connection():
            raise ValueError("MySQL database connection failed")

        # Configure database connection using SQLAlchemy engine
        # 使用SQLDatabase包装SQLAlchemy引擎，并禁用示例行以保持Schema清洁
        self.db_engine = SQLDatabase(
            self.db_config.get_sqlalchemy_engine(),
            sample_rows_in_table_info=0
        )

        # Set up language models with specific configurations
        self.llm = ChatOpenAI(temperature=0)  # Default model (e.g., GPT-3.5)

        # 使用更新的参数，避免警告
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key=self.deepseek_api_key,
            temperature=0,
            max_retries=2,
            timeout=60.0,
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


class DiscoveryAgent:
    def __init__(self):
        # Initialize configuration and toolkit
        self.config = Config()
        self.toolkit = SQLDatabaseToolkit(db=self.config.db_engine, llm=self.config.llm)
        self.tools = self.toolkit.get_tools()

        # Set up the chat prompt and OpenAI-based agent
        self.chat_prompt = self.create_chat_prompt()
        self.agent = create_openai_functions_agent(
            llm=self.config.llm,
            prompt=self.chat_prompt,
            tools=self.tools
        )

        # Configure agent executor for query handling
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15
        )

    def run_query(self, q):
        # Execute a SQL query using the configured database engine
        return self.config.db_engine.run(q)

    def create_chat_prompt(self):
        # A general-purpose system message. The detailed instructions will be passed in the input.
        system_message = SystemMessagePromptTemplate.from_template(
            """
            You are an AI assistant that helps with database tasks.
            You must follow instructions precisely to format the output as a single, clean JSON object.
            """
        )

        # Define the human message template
        human_message = HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}")

        # Combine the system and human templates into a chat prompt
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def discover(self) -> nx.Graph:
        """Perform schema discovery by processing one table at a time."""
        logger.info("Performing discovery by fetching schema table by table...")

        # 1. Get all table names from the database
        all_table_names = self.config.db_engine.get_table_names()
        logger.info(f"Found {len(all_table_names)} tables. Processing one by one.")

        final_schema_data = []

        # 2. Loop through each table name to get its structure from the LLM
        for table_name in all_table_names:
            logger.info(f"Processing table: {table_name}")

            # a. Get the DDL for the single table
            table_ddl = self.config.db_engine.get_table_info(table_names=[table_name])

            # b. Create a specific, small prompt for this table
            prompt = f"""
            Based on the following MySQL DDL for the table `{table_name}`, generate a JSON object representing its structure.

            Table Schema:
            ```sql
            {table_ddl}
            ```

            Instructions:
            Create a single JSON object for the table. The object should have keys "tableName" and "columns".
            Each object in the "columns" array should have "columnName", "columnType", "isOptional", and "foreignKeyReference".
            For "isOptional", use `false` if "NOT NULL" is present, otherwise `true`.
            For "foreignKeyReference", set it to `null` for now.

            ## Mandatory Rules:
            - Only output the final JSON object.
            - Do not include any extra commentary or markdown formatting.
            """

            try:
                # c. Invoke the agent with the small, focused prompt
                response = self.agent_executor.invoke({"input": prompt})
                
                # d. Parse the response for this single table
                table_data = self.parseJson(response['output'])
                final_schema_data.append(table_data)
                logger.info(f"Successfully processed table: {table_name}")

            except (ValueError, json.JSONDecodeError) as e:
                # If one table fails, log it and continue with the rest
                logger.error(f"Failed to process table '{table_name}'. Skipping. Error: {e}")
                continue
        
        # 3. Discover relationships based on naming conventions
        logger.info("Heuristically discovering relationships between tables...")
        schema_with_relations = self.discover_relationships(final_schema_data)

        # 4. After the loop, create the graph from the accumulated and enriched data
        logger.info("All tables processed. Creating graph...")
        graph = self.create_graph_from_data(schema_with_relations)
        return graph

    def discover_relationships(self, schema_data: List[dict]) -> List[dict]:
        """
        Analyzes schema data to heuristically find relationships based on naming conventions.
        Handles cases like 'staff_id' -> 'tb_staff_info.id'.
        """
        entity_to_table_map = {}
        table_names = [table['tableName'] for table in schema_data]
        
        # Sort by length to give precedence to shorter names (e.g., 'tb_staff' over 'tb_staff_info')
        table_names.sort(key=len)

        for name in table_names:
            if not name.startswith('tb_'):
                continue
            
            base_entity = name.replace('tb_', '')  # e.g., 'staff_info'
            
            # Always map the full entity name
            entity_to_table_map[base_entity] = name
            
            # If it's a multi-word entity, also map the first word if not already taken
            parts = base_entity.split('_')
            if len(parts) > 1:
                short_entity = parts[0]  # e.g., 'staff'
                if short_entity not in entity_to_table_map:
                    entity_to_table_map[short_entity] = name

        # Create a map of table names to their presumed primary key ('id')
        pk_map = {
            table['tableName']: 'id' for table in schema_data 
            if any(col['columnName'] == 'id' for col in table['columns'])
        }

        # Iterate through each table and column to find potential foreign keys
        for table in schema_data:
            table_name = table['tableName']
            for column in table['columns']:
                col_name = column['columnName']

                # Rule 1: Check for '..._id' suffix, e.g., 'user_id' or 'staff_id'
                if col_name.endswith('_id'):
                    entity_name = col_name[:-3]  # Remove '_id' to get 'user' or 'staff'
                    
                    # Find the corresponding full table name using our enhanced map
                    ref_table_name = entity_to_table_map.get(entity_name)
                    
                    if ref_table_name and ref_table_name in pk_map:
                        ref_pk = pk_map[ref_table_name]
                        column['foreignKeyReference'] = {
                            "table": ref_table_name,
                            "column": ref_pk
                        }
                        logger.info(f"Found relationship: {table_name}.{col_name} -> {ref_table_name}.{ref_pk}")
                        continue # Move to the next column

                # Rule 2: Special case for self-referencing 'parent_id'
                if col_name == 'parent_id' and table_name in pk_map:
                    ref_pk = pk_map[table_name]
                    column['foreignKeyReference'] = {
                        "table": table_name,
                        "column": ref_pk
                    }
                    logger.info(f"Found self-relationship: {table_name}.{col_name} -> {table_name}.{ref_pk}")

        return schema_data

    def parseJson(self, output_: str):
        """
        Parses a JSON string that might be wrapped in markdown.
        Returns a Python object (list or dict).
        """
        match = re.search(r"```json\s*([\s\S]*?)\s*```", output_, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            json_str = output_.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse the following string as JSON: '{json_str}'")
            raise

    def create_graph_from_data(self, data: List[dict]):
        """
        Converts a list of table schema dictionaries into a NetworkX graph.
        """
        G = nx.Graph()
        nodeIds = 0
        columnIds = len(data) + 1
        labeldict = {}
        canonicalColumns = {}

        for table in data:
            nodeIds += 1
            G.add_node(nodeIds)
            G.nodes[nodeIds]['tableName'] = table["tableName"]
            labeldict[nodeIds] = table["tableName"]

            for column in table["columns"]:
                columnIds += 1
                G.add_node(columnIds)
                G.nodes[columnIds]['columnName'] = column["columnName"]
                G.nodes[columnIds]['columnType'] = column["columnType"]
                G.nodes[columnIds]['isOptional'] = column["isOptional"]
                labeldict[columnIds] = column["columnName"]
                canonicalColumns[table["tableName"] + column["columnName"]] = columnIds
                G.add_edge(nodeIds, columnIds)

        for table in data:
            for column in table["columns"]:
                if column.get("foreignKeyReference") is not None:
                    this_column = table["tableName"] + column["columnName"]
                    ref = column["foreignKeyReference"]
                    if ref and ref.get("table") and ref.get("column"):
                        reference_column_ = ref["table"] + ref["column"]
                        if this_column in canonicalColumns and reference_column_ in canonicalColumns:
                            G.add_edge(canonicalColumns[this_column], canonicalColumns[reference_column_])

        return G

    def plot_graph(self, G, title="Graph Visualization"):
        """Plot a NetworkX graph with specific colors for tables and fields."""
        labels = {node: G.nodes[node].get('tableName') or G.nodes[node].get('columnName') or str(node) for node in G.nodes}
        
        color_map = []
        for node in G.nodes:
            if 'tableName' in G.nodes[node]:
                color_map.append('red')
            elif 'columnName' in G.nodes[node]:
                color_map.append('green')
            else:
                color_map.append('blue')

        pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
        plt.rcParams['figure.figsize'] = [20, 20]
        nx.draw(G, pos, labels=labels, node_color=color_map, with_labels=True)
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    try:
        patch_all_clients()
        agent = DiscoveryAgent()
        G = agent.discover()
        if G.number_of_nodes() > 0:
            agent.plot_graph(G)
        else:
            logger.warning("Graph is empty. Nothing to plot.")

    except Exception as e:
        logger.error(f"An unrecoverable error occurred during the discovery process: {e}", exc_info=True)
