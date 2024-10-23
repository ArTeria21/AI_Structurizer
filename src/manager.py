import os
from typing import Iterator

# Parsing text from documents
from unstructured.partition.auto import partition
from tika import parser
import pandas as pd
import docx

class Manager:
    avalable_for_processing_extensions = ('.txt', '.csv', '.pdf', '.md', '.docx')
    def __init__(self, input_folder: str, output_folder: str, agent, batch_size: int = 8000) -> None:
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.agent = agent
        self.batch_size = batch_size

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

    def extract_text(self, file_path: str) -> str:
        if not os.path.isfile(file_path):
            raise ValueError(f"'{file_path}' is not exists")
        
        if not file_path.endswith(self.avalable_for_processing_extensions):
            raise ValueError(f"'{file_path}' is not supported")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        text = ''

        try:
            if file_extension in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif file_extension == '.pdf':
                raw = parser.from_file(file_path)
                text = raw['content']
            elif file_extension in ['.docx']:
                doc = docx.Document(file_path)
                text = '\n'.join([para.text for para in doc.paragraphs])
            elif file_extension == '.csv':
                df = pd.read_csv(file_path)
                text = df.to_string()
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
                text = df.to_string()
            else:
                elements = partition(filename=file_path)
                text = '\n'.join([str(el) for el in elements])
        except Exception as e:
            raise ValueError(f"Error processing file {file_path}: {e}")

        return text
    
    def get_list_of_files(self) -> list:
        return [self.input_folder + '/' + f for f in os.listdir(self.input_folder) 
                if not f.startswith('.') 
                and f.endswith(self.avalable_for_processing_extensions)]
    
    def iterate_by_file(self, file_path: str) -> Iterator:
        text = self.extract_text(file_path)
        for i in range(0, len(text), self.batch_size):
            yield text[i:i+self.batch_size]
