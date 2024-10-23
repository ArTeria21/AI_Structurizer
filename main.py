import os
import re
import time
import unicodedata
from collections import defaultdict

from unstructured.partition.auto import partition
from tika import parser
import pandas as pd
import docx

from langchain.prompts import PromptTemplate
from langchain_together import Together

# Paths to input and output folders
input_folder = 'PATH_TO_INPUT_FOLDER'
output_folder = 'PATH_TO_OUTPUT_FOLDER'
together_api = 'YOUR_TOKEN'

# Initialize LLM with specified parameters
llm = Together(
    api_key=together_api,
    model="Qwen/Qwen2.5-72B-Instruct-Turbo",
    temperature=0.1,
    max_tokens=6000,
    repetition_penalty=1,
)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def extract_text(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    text = ''

    try:
        if file_extension in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file_extension == '.pdf':
            raw = parser.from_file(file_path)
            text = raw['content']
        elif file_extension in ['.docx', '.doc']:
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
        print(f"Error processing file {file_path}: {e}")

    return text

def normalize_topic_name(name):
    name = name
    name = unicodedata.normalize('NFKD', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.strip()
    return name

class LLMAgent:
    def __init__(self, llm):
        self.llm = llm
        self.existing_topics = set()
        self.template = """
You are an expert in word processing. Please read this text carefully and divide it into two topics. Write a separate text in Markdown format for each topic.
Write in detail about each topic, without simplifying the text. Write it in such a way that a person who is not familiar with this topic will understand everything.
Shorten the text to the most important information. Use as little text as possible to convey as much information as possible.

I also want you to create a graph of the notes. For this you can add links to the related notes using [[]] syntax.

Requirements:
Select *no more than two topics*. Don't use shared themes.
The topics should be different. For example, topics "machine learning", "getting started with machine learning" and "introduction to machine learning" are the same and should be avoided!
Topic names should be short and clear. Don't use abstract names. Use not more then four words.
Each topic should contain one word or concept. Define this word or concept in the context of the topic.
Write a separate Markdown document for each topic.
Use headings (#), lists(*), and other Markdown elements.
At the end of the document, list the key topics and terms.
If you find it helpful, add links to other documents in this format [[title of document]].
Write texts in *English only* even if the original language is not English.

USE THE POPULAR WIKI MARKDOWN SYNTAX!
Notice that in your response you should include texts for all the topics. It is very important to not write cutted parts of the text.

- There is a list of existing documents you can use for links and references. Choose only from them. Don't add new ones:
{existing_topics}

Return the text in the format:

---
Topic: Topic name 1

# Topic name 1

(Conspect of the topic in Markdown)

---
Topic: Topic name 2

# Topic name 2

(Conspect of the topic in Markdown)

---

Text to structure:
"{text}"
"""
        self.prompt = PromptTemplate(template=self.template, input_variables=["text", "existing_topics"])

    def process_text(self, text):
        existing_topics_str = '\n'.join(f'- {topic}' for topic in self.existing_topics)
        chain = self.prompt | self.llm
        result = chain.invoke({"text": text, "existing_topics": existing_topics_str})
        return result

    def add_topics(self, new_topics):
        self.existing_topics.update(new_topics)

# Dictionaries to store topics and related documents
doc_topics = {}
topic_docs = defaultdict(set)

# Maximum length of text to send to the model (in characters)
MAX_INPUT_LENGTH = 6000  # Adjust according to your model's limitations
MIN_RESPONSE_LENGTH = 1200

# Initialize the LLM agent
agent = LLMAgent(llm=llm)

# Process files in the folder
for root, dirs, files in os.walk(input_folder):
    for filename in files:
        file_path = os.path.join(root, filename)
        print(f"Processing file: {file_path}")
        text = extract_text(file_path)

        if text:
            # Split text into parts if it's too long
            text_parts = [text[i:i+MAX_INPUT_LENGTH] for i in range(0, len(text), MAX_INPUT_LENGTH)]
            for part in text_parts:
                # Add a delay to avoid exceeding request limits
                time.sleep(1.1)  # Wait slightly more than 1 second between requests

                try:
                    result = agent.process_text(part)
                    if len(result) < MIN_RESPONSE_LENGTH:
                        print(f"Model response is too short. Skipping. Response: \n{result}")
                        continue

                    if not result or len(result.strip()) == 0:
                        print(f"Empty response from the model for a part of the text from {file_path}. Skipping.")
                        continue

                    pattern = r"---\s*Topic:\s*(.*?)\n(.*?)(?=\n---|$)"
                    matches = re.findall(pattern, result, re.DOTALL)

                    if not matches:
                        print(f"Couldn't find topics in the model's response for file {file_path}.")
                        continue

                    for idx, (topic_name, content) in enumerate(matches):
                        if re.match(r'topic name \d+', topic_name.strip().lower()):
                            print(f"Incorrect topic name '{topic_name}' detected. Skipping.")
                            continue

                        topic_name_normalized = normalize_topic_name(topic_name)
                        safe_topic_name = re.sub(r'[\\/*?:"<>|]', "_", topic_name_normalized)
                        file_name = f"{safe_topic_name}.md"
                        file_output_path = os.path.join(output_folder, file_name)

                        key_topics = re.findall(r"Key topics:\s*(.*)", content)
                        if key_topics:
                            topics = [topic.strip() for topic in key_topics[0].split(',')]
                            topics = list(set(topics))
                        else:
                            topics = []

                        if file_name in doc_topics:
                            doc_topics[file_name].extend(topics)
                            doc_topics[file_name] = list(set(doc_topics[file_name]))
                        else:
                            doc_topics[file_name] = topics

                        for topic in topics:
                            topic_docs[topic].add(file_name)

                        agent.add_topics([topic_name])
                        print(agent.existing_topics)
                        # Save the file
                        with open(file_output_path, 'w', encoding='utf-8') as f:
                            f.write(content.strip() + '\n')
                except ValueError as e:
                    print(f"Error processing a part of the text from {file_path}: {e}")
                    continue

print("Processing completed!")
