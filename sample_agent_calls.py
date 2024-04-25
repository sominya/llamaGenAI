from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.llms.openai import OpenAI

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core import Settings
import os

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"



Settings.llm = OpenAI(temperature=0.2, model="gpt-3.5-turbo")

# mkdir -p 'data/10q/'
# wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_march_2022.pdf' -O 'data/10q/uber_10q_march_2022.pdf'
# wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_june_2022.pdf' -O 'data/10q/uber_10q_june_2022.pdf'
# wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10q/uber_10q_sept_2022.pdf' -O 'data/10q/uber_10q_sept_2022.pdf'

# Load Data
march_2022 = SimpleDirectoryReader(
    input_files=["./data/10q/uber_10q_march_2022.pdf"]
).load_data()
june_2022 = SimpleDirectoryReader(
    input_files=["./data/10q/uber_10q_june_2022.pdf"]
).load_data()
sept_2022 = SimpleDirectoryReader(
    input_files=["./data/10q/uber_10q_sept_2022.pdf"]
).load_data()

# Build Indices
march_index = VectorStoreIndex.from_documents(march_2022)
june_index = VectorStoreIndex.from_documents(june_2022)
sept_index = VectorStoreIndex.from_documents(sept_2022)

# Build Query Engine
march_engine = march_index.as_query_engine(similarity_top_k=3)
june_engine = june_index.as_query_engine(similarity_top_k=3)
sept_engine = sept_index.as_query_engine(similarity_top_k=3)

# Build List of query engines
query_engine_tools = [
    QueryEngineTool(
        query_engine=sept_engine,
        metadata=ToolMetadata(
            name="sept_22",
            description=(
                "Provides information about Uber quarterly financials ending"
                " September 2022"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=june_engine,
        metadata=ToolMetadata(
            name="june_22",
            description=(
                "Provides information about Uber quarterly financials ending"
                " June 2022"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=march_engine,
        metadata=ToolMetadata(
            name="march_22",
            description=(
                "Provides information about Uber quarterly financials ending"
                " March 2022"
            ),
        ),
    ),
]

# Buld Final Chained Query engine
s_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools
)