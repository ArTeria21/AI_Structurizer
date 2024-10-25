import os
import logging
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers.fix import OutputFixingParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchain_together import Together

class Topic(BaseModel):
    topics: str = Field(description='Top 3 main topics of the text separated by Semicolon (;)')

topic_parser = JsonOutputParser(pydantic_object=Topic)

class Note(BaseModel):
    title: str = Field(description='Title of the note')
    note_text: str = Field(description='The text of the note in markdown format')

note_parser = JsonOutputParser(pydantic_object=Note)

# Initialize log
logger = logging.getLogger(__name__)

def load_prompt(file_path: str) -> str:
    if not os.path.isfile(file_path):
        logger.error(f"Prompt file not found: {file_path}")
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Load prompts from external files
get_topics_prompt_text = load_prompt(os.path.join(os.path.dirname(__file__), 'prompts', 'get_topics_prompt.txt'))
get_notes_prompt_text = load_prompt(os.path.join(os.path.dirname(__file__), 'prompts', 'get_notes_prompt.txt'))
fix_note_prompt_text = load_prompt(os.path.join(os.path.dirname(__file__), 'prompts', 'fix_prompt.txt'))
fix_topic_prompt_text = load_prompt(os.path.join(os.path.dirname(__file__), 'prompts', 'fix_prompt.txt'))

get_topics_prompt = PromptTemplate(
    template=get_topics_prompt_text,
    input_variables=['text'],
    partial_variables={'format_instructions': topic_parser.get_format_instructions()}
)

get_notes_prompt = PromptTemplate(
    template=get_notes_prompt_text,
    input_variables=['topic', 'text', 'existing_topics'],
    partial_variables={'format_instructions': note_parser.get_format_instructions()}
)

fix_note_prompt = PromptTemplate(
    template=fix_note_prompt_text,
    input_variables=['completion', 'error'],
    partial_variables={'format_instructions': note_parser.get_format_instructions()}
)

fix_topic_prompt = PromptTemplate(
    template=fix_topic_prompt_text,
    input_variables=['completion', 'error'],
    partial_variables={'format_instructions': topic_parser.get_format_instructions()}
)

# Fixing wrong formatting
fixer_llm = Together(
    api_key=os.getenv('API_KEY'), 
    model='Qwen/Qwen2.5-72B-Instruct-Turbo',
    temperature=0.05, 
    max_tokens=3500, 
    repetition_penalty=1.2
)

fixer_note_parser = OutputFixingParser.from_llm(
    fixer_llm, 
    parser=note_parser, 
    max_retries=3, 
    prompt=fix_note_prompt
)

fixer_topic_parser = OutputFixingParser.from_llm(
    fixer_llm, 
    parser=topic_parser, 
    max_retries=3, 
    prompt=fix_topic_prompt
)
