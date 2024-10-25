import os
import logging
from dotenv import load_dotenv
from .prompts import (
    get_topics_prompt, 
    fix_topic_prompt, 
    get_notes_prompt, 
    fix_note_prompt,
    fixer_topic_parser,
    fixer_note_parser
)

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# Interaction with LLM
from langchain_together import Together

load_dotenv()

class Agent:
    def __init__(self, output_folder: str) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Initializing Agent")
        api_token = os.getenv('API_KEY')
        if not api_token:
            self.logger.error("API_KEY not found in environment variables")
            raise ValueError("API_KEY not found in environment variables")
        
        self.topics_llm = Together(
            api_key=api_token, 
            model='Qwen/Qwen2.5-72B-Instruct-Turbo', 
            temperature=0.3, 
            max_tokens=1024, 
            repetition_penalty=1
        )
        
        self.writing_llm = Together(
            api_key=api_token, 
            model='meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo',
            temperature=0.1, 
            max_tokens=4000,
            repetition_penalty=1.1
        )
        
        self.output_folder = output_folder
        self.existing_titles = self.get_existing_titles()
        self.logger.debug("Agent initialized successfully")
    
    def get_existing_titles(self) -> set:
        self.logger.debug("Fetching existing titles from output folder")
        existing_titles = set()
        if os.path.exists(self.output_folder):
            for file_name in os.listdir(self.output_folder):
                if file_name.endswith('.md'):
                    title = os.path.splitext(file_name)[0]
                    existing_titles.add(title)
        self.logger.debug(f"Found {len(existing_titles)} existing titles")
        return existing_titles

    def get_text_topics(self, text: str) -> list:
        self.logger.debug("Getting text topics")
        try:
            # Формируем промпт
            prompt = get_topics_prompt.format(text=text)
            self.logger.debug(f"Formatted get_topics_prompt: {prompt}")
            
            # Отправляем запрос в LLM
            response = self.topics_llm(prompt)
            self.logger.debug(f"LLM response for topics: {response}")
            
            # Парсим и исправляем ответ
            fixed_response = fixer_topic_parser.parse(response)
            self.logger.debug(f"Fixed topics response: {fixed_response}")
            
            # Обрабатываем результат
            topics = fixed_response['topics'].split(';')
            topics = [topic.strip() for topic in topics]
            self.logger.info(f"Extracted topics: {topics}")
            return topics
        except Exception as e:
            self.logger.error(f"Error getting topics: {e}")
            raise

    def write_abstract(self, topic: str, text: str) -> dict:
        self.logger.debug(f"Writing abstract for topic: {topic}")
        try:
            existing_topics_str = '; '.join(self.existing_titles)
            # Формируем промпт
            prompt = get_notes_prompt.format(topic=topic, text=text, existing_topics=existing_topics_str)
            self.logger.debug(f"Formatted get_notes_prompt: {prompt}")
            
            # Отправляем запрос в LLM
            response = self.writing_llm(prompt)
            self.logger.debug(f"Unparsed LLM response for note with topic '{topic}': {response}")
            
            # Парсим и исправляем ответ
            fixed_response = fixer_note_parser.parse(response)
            self.logger.debug(f"parsed note response: {fixed_response}")
            self.logger.info(f'written note for topic {topic}: {fixed_response["note_text"][:100]}...')
            return fixed_response
        except Exception as e:
            self.logger.error(f"Error writing abstract for topic '{topic}': {e}")
            raise

    def process_text(self, text: str) -> None:
        self.logger.debug("Processing text chunk")
        try:
            topics = self.get_text_topics(text)
            for topic in topics:
                result = self.write_abstract(topic, text)
                # result is a dictionary с 'title' и 'note_text'
                title = result['title']
                note_text = result['note_text']
                
                # Обновляем существующие названия
                self.existing_titles.add(title)
                
                # Сохраняем каждую заметку в отдельный Markdown файл, если она еще не существует
                note_file_name = f"{title}.md"
                note_file_path = os.path.join(self.output_folder, note_file_name)
                if not os.path.exists(note_file_path):
                    with open(note_file_path, 'w', encoding='utf-8') as f:
                        f.write(note_text)
                    self.logger.info(f"Saved new note: {note_file_path}")
                else:
                    self.logger.info(f"Note already exists: {note_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to process text chunk: {e}")
