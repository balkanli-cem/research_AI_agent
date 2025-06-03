import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, saving_tool

load_dotenv()
# Define a Pydantic model for the output
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0.7,
#     max_tokens=1000,
# )
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,
    max_tokens=8192,
)
# response = llm.invoke("What is meaning of life?")
# print(response)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """
         You are a research assistant that will help generate a research paper.
         Answer the user query and use necessary tools.
         Wrap the output in this format and provide no other text\n{format_instructions}
         """
         ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, saving_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)
query = input("What can I help you research about?")
raw_response = agent_executor.invoke(
    {
        "query": query,
    })

# Run the LLM with the prompt

print(raw_response)

try:
    structed_response = parser.parse(raw_response.get("output")[0]["text"])
    print("Structured Response:", structed_response)
except Exception as e:
    print("Error parsing response:", e, "Raw response:", raw_response)
