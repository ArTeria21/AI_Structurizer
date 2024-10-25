import os
import logging
from typing import Iterator

# Preprocessing text
import nltk
from nltk.tokenize import sent_tokenize
from translatepy.translators.yandex import YandexTranslate

# Parsing text from documents
from unstructured.partition.auto import partition
import fitz 
import pandas as pd
import docx

# Interaction with LLM
from .llm_agent import Agent

class Manager:
    available_for_processing_extensions = ('.txt', '.csv', '.pdf', '.md', '.docx')
    
    def __init__(self, input_folder: str, output_folder: str, batch_size: int = 12000) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Initializing Manager")
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.batch_size = batch_size
        self.agent = Agent(output_folder=self.output_folder)
        nltk.download('punkt', quiet=True)  # Загружаем необходимые данные для токенизации
        self.logger.debug("Manager initialized successfully")
        self.translator = YandexTranslate()
    
    @property
    def input_folder(self) -> str:
        return self._input_folder
    
    @input_folder.setter
    def input_folder(self, path: str) -> None:
        if not os.path.isdir(path):
            raise ValueError(f"'{path}' is not a valid directory")
        self._input_folder = path
    
    @property
    def output_folder(self) -> str:
        return self._output_folder
    
    @output_folder.setter
    def output_folder(self, path: str) -> None:
        if not os.path.isdir(path):
            os.mkdir(path)
        self._output_folder = path

    def translate_text_to_english(self, text: str) -> str:
        translated_text = ''
        sentance_amount = 6
        # Разделим текст на предложения
        sentences = text.split('. ')
        
        for start in range(0, len(sentences), sentance_amount):
            chunk = ' '.join(sentences[start:start + sentance_amount])
            
            try:
                translated_chunk = str(self.translator.translate(chunk, destination_language='EN'))
                if translated_chunk:
                    translated_text += translated_chunk + ' '
                else:
                    self.logger.warning(f"Translation returned empty for chunk: {chunk}")
            except Exception as e:
                self.logger.error(f"Error during translation of chunk: {e}")
                continue
        
        if not translated_text.strip():
            self.logger.error("Translation returned empty text.")
            raise ValueError("Failed to translate text to English")
        return translated_text.strip()
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        text = ''
        try:
            with fitz.open(file_path) as pdf:
                for page_num in range(pdf.page_count):
                    page = pdf[page_num]
                    text += page.get_text("text")  # Извлечение текста в виде строки
        except Exception as e:
            self.logger.error(f"Failed to extract text from PDF {file_path}: {e}")
            raise ValueError(f"Error processing file {file_path}: {e}")
        return text
    
    def extract_text(self, file_path: str) -> str:
        self.logger.debug(f"Extracting text from {file_path}")
        if not os.path.isfile(file_path):
            raise ValueError(f"'{file_path}' does not exist")
        
        if not file_path.endswith(self.available_for_processing_extensions):
            raise ValueError(f"'{file_path}' is not supported")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        text = ''

        try:
            if file_extension in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif file_extension == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_extension == '.docx':
                doc = docx.Document(file_path)
                text = '\n'.join([para.text for para in doc.paragraphs])
            elif file_extension == '.csv':
                df = pd.read_csv(file_path)
                text = df.to_string()
            else:
                elements = partition(filename=file_path)
                text = '\n'.join([str(el) for el in elements])
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            raise ValueError(f"Error processing file {file_path}: {e}")
        
        self.logger.debug(f"Extracted text length from {file_path}: {len(text)} characters")
        return self.translate_text_to_english(text)
    
    def get_list_of_files(self) -> list:
        self.logger.debug("Listing files for processing")
        return [
            os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder) 
            if not f.startswith('.') 
            and f.endswith(self.available_for_processing_extensions)
        ]
    
    def iterate_by_file(self, file_path: str) -> Iterator[str]:
        self.logger.debug(f"Iterating over file: {file_path}")
        text = self.extract_text(file_path)
        sentences = sent_tokenize(text)
        current_chunk = ''
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.batch_size:
                if current_chunk:
                    current_chunk += ' ' + sentence
                else:
                    current_chunk = sentence
            else:
                yield current_chunk.strip()
                current_chunk = sentence
        if current_chunk:
            yield current_chunk.strip()
    
    def process_files(self) -> None:
        self.logger.info("Starting to process files")
        files = self.get_list_of_files()
        if not files:
            self.logger.warning("No files found for processing")
            return
        for file_path in files:
            self.logger.info(f"Processing file: {file_path}")
            try:
                for chunk in self.iterate_by_file(file_path):
                    self.logger.debug(f"Processing chunk of size {len(chunk)}")
                    self.agent.process_text(text=chunk)
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
