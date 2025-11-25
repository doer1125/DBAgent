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
import jieba

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
# Suppress jieba logging output
logging.getLogger("jieba").setLevel(logging.WARNING)


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
            Create a single JSON object for the table. The object should have keys "tableName", "description", and "columns".
            Extract the table comment as the "description". If no comment is available, set "description" to null.
            Each object in the "columns" array should have "columnName", "columnType", "isOptional", "columnDescription", and "foreignKeyReference".
            Extract the column comment as "columnDescription". If no comment is available, set "columnDescription" to null.
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
            G.nodes[nodeIds]['description'] = table.get("description")
            labeldict[nodeIds] = table["tableName"]

            for column in table["columns"]:
                columnIds += 1
                G.add_node(columnIds)
                G.nodes[columnIds]['columnName'] = column["columnName"]
                G.nodes[columnIds]['columnType'] = column["columnType"]
                G.nodes[columnIds]['isOptional'] = column["isOptional"]
                G.nodes[columnIds]['columnDescription'] = column.get("columnDescription") # Add column description
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


class InferenceAgent:
    def __init__(self):
        # Initialize configuration, toolkit, and tools
        self.config = Config()
        self.toolkit = SQLDatabaseToolkit(db=self.config.db_engine, llm=self.config.llm)
        self.tools = self.toolkit.get_tools()
        self.chat_prompt = self.create_chat_prompt()

        # Create an OpenAI-based agent with tools and prompt
        self.agent = create_openai_functions_agent(
            llm=self.config.llm,
            prompt=self.chat_prompt,
            tools=self.tools
        )

        # Configure the agent executor with runtime settings
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=15
        )

        # Test the database connection
        self.test_connection()

    def test_connection(self):
        # Verify the database connection by running a test query
        try:
            self.show_tables()
            logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise

    def show_tables(self) -> str:
        # Query to list all tables and views in the MySQL database
        q = '''
            SELECT TABLE_NAME as name,
                   TABLE_TYPE as type
            FROM information_schema.tables
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_TYPE IN ('BASE TABLE', 'VIEW');
            '''
        return self.run_query(q)

    def run_query(self, q: str) -> str:
        # Execute a SQL query and handle errors if they occur
        try:
            return self.config.db_engine.run(q)
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return f"Error executing query: {str(e)}"

    def create_chat_prompt(self) -> ChatPromptTemplate:
        # Create a system prompt to guide the LLM's behavior and response format
        system_message = SystemMessagePromptTemplate.from_template(
            """You are a database inference expert for a mysql database named {db_name}.
            Your job is to answer questions by querying the database and providing clear, accurate results.

            Rules:
            1. ONLY execute queries that retrieve data
            2. DO NOT provide analysis or recommendations
            3. Format responses as:
               Query Executed: [the SQL query used]
               Results: [the query results]
               Summary: [brief factual summary of the findings]
            4. Keep responses focused on the data only
            """
        )

        # Create a template for user-provided input
        human_message = HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}")

        # Combine system and human message templates into a chat prompt
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def analyze_question_with_graph(self, db_graph: nx.Graph, question: str) -> dict:
        """
        Analyzes the user's question against the database graph to find relevant tables and columns.
        It tokenizes Chinese questions and matches keywords against table names, table descriptions,
        column names, and column descriptions.
        """
        logger.info(f"Starting graph analysis for question: '{question}'")

        # Use jieba to tokenize the question for keyword matching
        try:
            question_keywords = set(jieba.cut(question.lower()))
            # Remove short/common words
            question_keywords = {kw for kw in question_keywords if len(kw.strip()) > 1}
        except ImportError:
            logger.warning("jieba is not installed. Falling back to simple whitespace tokenization. Please run 'pip install jieba'.")
            question_keywords = set(question.lower().split())

        analysis = {
            'tables': [],
            'relationships': [],
            'columns': [],
            'possible_paths': []
        }
        
        matched_tables = []

        # Iterate through all nodes to find tables
        for node_id, node_data in db_graph.nodes(data=True):
            if 'tableName' not in node_data:
                continue

            table_name = node_data['tableName'].lower()
            table_description = (node_data.get('description') or "").lower()
            
            # Collect all column names and descriptions for this table to aid in scoring
            column_search_text = []
            for neighbor in db_graph.neighbors(node_id):
                col_data = db_graph.nodes[neighbor]
                if 'columnName' in col_data:
                    column_search_text.append(col_data['columnName'].lower())
                    if col_data.get('columnDescription'):
                        column_search_text.append(col_data['columnDescription'].lower())
            
            # Combine table name, table description, and all column info for initial relevance scoring
            search_text = table_name + " " + table_description + " " + " ".join(column_search_text)
            
            # Calculate a relevance score
            score = 0
            for keyword in question_keywords:
                if keyword in search_text:
                    # Give higher weight to matches in the table name or description
                    if keyword in table_name or keyword in table_description:
                        score += 2
                    else: # Match in column name or description
                        score += 1

            if score > 0:
                matched_tables.append({
                    'score': score, 
                    'node_id': node_id, 
                    'node_data': node_data
                })

        # Sort tables by relevance score in descending order
        matched_tables.sort(key=lambda x: x['score'], reverse=True)
        
        # Process the most relevant tables
        for match in matched_tables:
            node_data = match['node_data']
            node_id = match['node_id']
            
            logger.info(f"Found relevant table: {node_data['tableName']} (Score: {match['score']})")
            
            table_info = {'name': node_data['tableName'], 'columns': []}

            # Find relevant columns in this table by checking against question keywords
            for neighbor in db_graph.neighbors(node_id):
                col_data = db_graph.nodes[neighbor]
                if 'columnName' in col_data:
                    col_name = col_data['columnName'].lower()
                    col_description = (col_data.get('columnDescription') or "").lower()
                    
                    # Match keywords against column name AND column description
                    if any(kw in col_name or kw in col_description for kw in question_keywords):
                        table_info['columns'].append({
                            'name': col_data['columnName'],
                            'type': col_data['columnType'],
                            'table': node_data['tableName'],
                            'description': col_data.get('columnDescription') # Include column description in analysis output
                        })
                        logger.info(f"  -> Found relevant column: {col_data['columnName']} (Description: {col_data.get('columnDescription')})")

            analysis['tables'].append(table_info)

        return analysis

    def query(self, text: str, db_graph) -> str:
        # Execute a query using graph-based analysis or standard prompt
        try:
            if db_graph:
                logger.info(f"Analyzing query with graph: '{text}'")

                # Analyze the question with the database graph
                graph_analysis = self.analyze_question_with_graph(db_graph, text)
                logger.info(f"Graph Analysis Results:\n{json.dumps(graph_analysis, indent=2)}")

                # Enhance the prompt with graph analysis context
                enhanced_prompt = f"""
                Database Structure Analysis:
                - Available Tables: {[t['name'] for t in graph_analysis['tables']]}
                - Table Relationships: {graph_analysis['possible_paths']}
                - Relevant Columns: {[{'table': col['table'], 'name': col['name'], 'description': col.get('description')} for t in graph_analysis['tables'] for col in t['columns']]}

                User Question: {text}

                Use this structural information to form an accurate query.
                """
                logger.info("Enhanced prompt created with graph context")
                return self.agent_executor.invoke({"input": enhanced_prompt, "db_name": self.config.db})['output']

            logger.info(f"No graph available, executing standard query: '{text}'")
            return self.agent_executor.invoke({"input": text, "db_name": self.config.db})['output']

        except Exception as e:
            # Handle errors during query processing
            logger.error(f"Error in inference query: {str(e)}", exc_info=True)
            return f"Error processing query: {str(e)}"


class PlannerAgent:
    def __init__(self):
        # Initialize configuration and planner prompt
        self.config = Config()
        self.planner_prompt = self.create_planner_prompt()

    def create_planner_prompt(self):
        # Define the system template for planning instructions
        system_template = """You are a friendly planning agent that creates specific plans to answer questions about THIS database only.

        Available actions:
        1. Inference: [query] - Use this prefix for database queries
        2. General: [response] - Use this prefix for friendly responses

        Create a SINGLE, SEQUENTIAL plan where:
        - Each step should be exactly ONE line
        - Each step must start with either 'Inference:' or 'General:'
        - Steps must be in logical order
        - DO NOT repeat steps
        - Keep the plan minimal and focused

        Example format:
        Inference: Get all artists from the database
        Inference: Count tracks per artist
        General: Provide the results in a friendly way
        """

        # Define the human message template for user input
        human_template = "Question: {question}\n\nCreate a focused plan with appropriate action steps."

        # Combine system and human message templates into a chat prompt
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    def create_plan(self, question: str) -> list:
        # Generate a step-by-step plan to answer the given question
        try:
            logger.info(f"Creating plan for question: {question}")
            response = self.config.llm.invoke(self.planner_prompt.format(
                question=question
            ))

            # Extract and clean valid steps from the response
            steps = [step.strip() for step in response.content.split('\n')
                     if step.strip() and not step.lower() == 'plan:']

            # Provide a fallback message if no steps are returned
            if not steps:
                return ["General: I'd love to help you explore the database! What would you like to know?"]

            return steps

        except Exception as e:
            # Log and handle errors during plan creation
            logger.error(f"Error creating plan: {str(e)}", exc_info=True)
            return ["General: Error occurred while creating plan"]


def db_graph_reducer():
    # Reducer function for handling database graph updates
    def _reducer(previous_value: Optional[nx.Graph], new_value: nx.Graph) -> nx.Graph:
        if previous_value is None:  # If no previous graph exists, use the new graph
            return new_value
        return previous_value  # Otherwise, retain the existing graph
    return _reducer

def plan_reducer():
    # Reducer function for updating plans
    def _reducer(previous_value: Optional[List[str]], new_value: List[str]) -> List[str]:
        return new_value if new_value is not None else previous_value  # Use the new plan if available
    return _reducer

def classify_input_reducer():
    # Reducer function for input classification
    def _reducer(previous_value: Optional[str], new_value: str) -> str:
        return new_value  # Always replace with the latest classification
    return _reducer

class ConversationState(TypedDict):
    # Defines the conversation state structure and associated reducers
    question: str  # Current user question
    input_type: Annotated[str, classify_input_reducer()]  # Classification of the input type
    plan: Annotated[List[str], plan_reducer()]  # Step-by-step plan to respond to the question
    db_results: NotRequired[str]  # Optional field for database query results
    response: NotRequired[str]  # Optional field for generated response
    db_graph: Annotated[Optional[nx.Graph], db_graph_reducer()] = None  # Optional field for database graph


def classify_user_input(state: ConversationState) -> ConversationState:
    """Classifies user input to determine if it requires database access."""

    # Define a system prompt for classifying input into predefined categories
    system_prompt = """You are an input classifier. Classify the user's input into one of these categories:
    - DATABASE_QUERY: Questions about data, requiring database access
    - GREETING: General greetings, how are you, etc.
    - CHITCHAT: General conversation not requiring database
    - FAREWELL: Goodbye messages

    Respond with ONLY the category name."""

    # Prepare messages for the LLM, including the system prompt and user's input
    messages = [
        ("system", system_prompt),  # Instructions for the LLM
        ("user", state['question'])  # User's question for classification
    ]

    # Invoke the LLM with a zero-temperature setting for deterministic output
    # llm = ChatOpenAI(temperature=0)
    if 'config' not in state:
        state['config'] = Config()

    llm = state['config'].llm  # 使用已经配置好的 DeepSeek LLM
    response = llm.invoke(messages)
    classification = response.content.strip()  # Extract the category from the response

    # Log the classification result
    logger.info(f"Input classified as: {classification}")

    # Update the conversation state with the input classification
    return {
        **state,
        "input_type": classification
    }


class SupervisorAgent:
    def __init__(self):
        # Initialize configuration and agents
        self.config = Config()
        self.inference_agent = InferenceAgent()
        self.planner_agent = PlannerAgent()
        self.discovery_agent = DiscoveryAgent()

        # Prompts for different types of responses
        self.db_response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a response coordinator that creates final responses based on:
            Original Question: {question}
            Database Results: {db_results}

            Rules:
            1. ALWAYS include ALL results from database queries in your response
            2. Format the response clearly with each piece of information on its own line
            3. Use bullet points or numbers for multiple pieces of information
            """)
        ])

        self.chat_response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly AI assistant.
            Respond naturally to the user's message.
            Keep responses brief and friendly.
            Don't make up information about weather, traffic, or other external data.
            """)
        ])

    def create_plan(self, state: ConversationState) -> ConversationState:
        # Generate a plan using the PlannerAgent
        plan = self.planner_agent.create_plan(
            question=state['question']
        )

        # Log the plan, separating inference and general steps
        logger.info("Generated plan:")
        inference_steps = [step for step in plan if step.startswith('Inference:')]
        general_steps = [step for step in plan if step.startswith('General:')]

        if inference_steps:
            logger.info("Inference Steps:")
            for i, step in enumerate(inference_steps, 1):
                logger.info(f"  {i}. {step}")
        if general_steps:
            logger.info("General Steps:")
            for i, step in enumerate(general_steps, 1):
                logger.info(f"  {i}. {step}")

        return {
            **state,
            "plan": plan
        }

    def execute_plan(self, state: ConversationState) -> ConversationState:
        # Execute the generated plan step by step
        results = []

        try:
            for step in state['plan']:
                if ':' not in step:
                    continue

                step_type, content = step.split(':', 1)
                content = content.strip()

                if step_type.lower().strip() == 'inference':
                    # Handle inference steps using the InferenceAgent
                    try:
                        result = self.inference_agent.query(content, state.get('db_graph'))
                        results.append(f"Step: {step}\nResult: {result}")
                    except Exception as e:
                        logger.error(f"Error in inference step: {str(e)}", exc_info=True)
                        results.append(f"Step: {step}\nError: Query failed - {str(e)}")
                else:
                    # Handle general steps
                    results.append(f"Step: {step}\nResult: {content}")

            # Return state with results
            return {
                **state,
                "db_results": "\n\n".join(results) if results else "No results were generated."
            }

        except Exception as e:
            logger.error(f"Error in execute_plan: {str(e)}", exc_info=True)
            return {**state, "db_results": f"Error executing steps: {str(e)}"}

    def generate_response(self, state: ConversationState) -> ConversationState:
        # Generate the final response based on the input type
        logger.info("Generating final response")
        is_chat = state.get("input_type") in ["GREETING", "CHITCHAT", "FAREWELL"]
        prompt = self.chat_response_prompt if is_chat else self.db_response_prompt

        # Invoke the LLM to generate the response
        response = self.config.llm.invoke(prompt.format(
            question=state['question'],
            db_results=state.get('db_results', '')
        ))

        # Update state with the response and clear the plan
        return {**state, "response": response.content, "plan": []}

def discover_database(state: ConversationState) -> ConversationState:
    # Check if the database graph is already present in the state
    if state.get('db_graph') is None:
        logger.info("Performing one-time database schema discovery...")

        # Use the DiscoveryAgent to generate the database graph
        discovery_agent = DiscoveryAgent()
        graph = discovery_agent.discover()

        logger.info("Database schema discovery complete - this will be reused for future queries")

        # Update the state with the discovered database graph
        return {**state, "db_graph": graph}

    # Return the existing state if the database graph already exists
    return state



def create_graph():
    # Initialize the supervisor agent and state graph builder
    supervisor = SupervisorAgent()
    builder = StateGraph(ConversationState)

    # Add nodes representing processing steps in the flow
    builder.add_node("classify_input", classify_user_input)  # Classify the user input
    builder.add_node("discover_database", discover_database)  # Perform database discovery
    builder.add_node("create_plan", supervisor.create_plan)  # Create a plan based on input
    builder.add_node("execute_plan", supervisor.execute_plan)  # Execute the generated plan
    builder.add_node("generate_response", supervisor.generate_response)  # Generate the final response

    # Define the flow of states
    builder.add_edge(START, "classify_input")  # Start with input classification

    # Conditionally proceed to database discovery or directly to response generation
    builder.add_conditional_edges(
        "classify_input",
        lambda x: "discover_database" if x.get("input_type") == "DATABASE_QUERY" else "generate_response"
    )

    # Connect discovery to plan creation
    builder.add_edge("discover_database", "create_plan")

    # Conditionally execute the plan or generate a response if no plan exists
    builder.add_conditional_edges(
        "create_plan",
        lambda x: "execute_plan" if x.get("plan") is not None else "generate_response"
    )

    # Connect execution to response generation
    builder.add_edge("execute_plan", "generate_response")

    # End the process after generating the response
    builder.add_edge("generate_response", END)

    # Compile and return the state graph
    return builder.compile()


if __name__ == "__main__":
    try:
        patch_all_clients()
        # agent = DiscoveryAgent()
        # G = agent.discover()
        # if G.number_of_nodes() > 0:
        #     agent.plot_graph(G)
        # else:
        #     logger.warning("Graph is empty. Nothing to plot.")

        # Create the graph for processing
        graph = create_graph()

        state = graph.invoke({"question": "有多少个用户现在？"})
        print(f"State after second invoke: {state}")
        print(f"Response 2: {state['response']}\n")

    except Exception as e:
        logger.error(f"An unrecoverable error occurred during the discovery process: {e}", exc_info=True)
