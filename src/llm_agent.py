from dotenv import load_dotenv
from prompts import get_topics_prompt, topic_parser, get_notes_prompt, note_parser
import os

# interaction with LLM
from langchain_together import Together

load_dotenv()

class Agent:
    def __init__(self) -> None:
        api_token = os.getenv('API_KEY')
        self.topics_llm = Together(api_key=api_token, model='Qwen/Qwen2.5-72B-Instruct-Turbo', 
                            temperature=0.3, 
                            max_tokens=2000, 

                            repetition_penalty=1)
        
        self.writing_llm = Together(api_key=api_token, model='meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo',
                            temperature=0.1, 
                            max_tokens=3500,
                            repetition_penalty=1.1)
        
        self.existing_titles = set()
    
    def get_text_topics(self, text: str) -> list:
        chain = get_topics_prompt | self.topics_llm | topic_parser
        topics = chain.invoke(input={'text': text})
        topics = topics['topics'].split(';')
        return topics
    
    def write_abstract(self, topic: str, text: str) -> str:
        chain = get_notes_prompt | self.writing_llm | note_parser
        return chain.invoke(input={'topic': topic, 'text': text, 'existing_topics': str(self.existing_titles)})
    
    def process_text(self, text: str) -> list:
        notes = []
        topics = self.get_text_topics(text)
        print(topics)
        print('\n\n\n\n')
        for topic in topics:
            result = self.write_abstract(topic, text)
            notes.append(result)
        
        return notes


if __name__ == '__main__':
    agent = Agent()
    text_example = """
smthing
"""
    print(agent.process_text(text=text_example))