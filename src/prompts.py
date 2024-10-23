from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

class Topic(BaseModel):
    topics: str = Field(description='Top 3 main topics of the text separated by Semicolon (;)')

topic_parser = JsonOutputParser(pydantic_object=Topic)

get_topics_prompt = PromptTemplate(
    template="""
YOU ARE A TEXT ANALYSIS EXPERT, SPECIALIZED IN IDENTIFYING AND HIGHLIGHTING KEY TOPICS. YOU WILL RECEIVE A TEXT, AND YOUR TASK IS TO IDENTIFY **NO MORE THAN THREE DISTINCT TOPICS** DISCUSSED IN THE TEXT.

### INSTRUCTIONS:

1. **Identify up to three specific and non-overlapping key topics** discussed in the text.
2. **Ensure each topic** clearly describes the issues covered in the text and is **sufficiently specific**.
3. **Avoid general topic names** (e.g., "introduction to machine learning"). The topics should be **concise but comprehensive**.
4. **Always use English** for topic names, regardless of the text language.

### OUTPUT FORMAT:
- **List the topics, separated by semicolons**.
- Follow this exact output format, as it will be used for automatic processing:
```json
{format_instructions}
```

WHAT TO AVOID:
Do not provide more than three topics.
Do not use overlapping or general topic names.

Here is the text to analyze:
{text}
    """,
    input_variables=['text'],
    partial_variables={"format_instructions": topic_parser.get_format_instructions()}
)

# ---

class Note(BaseModel):
    title: str = Field(description='Title of the note')
    note_text: str = Field(description='The text of the note in markdown format')

note_parser = JsonOutputParser(pydantic_object=Note)

get_notes_prompt = PromptTemplate(
    template="""

```
YOU ARE A WORLD-CLASS OBSIDIAN NOTES WRITER AND KNOWLEDGE ORGANIZER, RENOWNED FOR YOUR ABILITY TO SUMMARIZE COMPLEX TEXTS INTO DETAILED ABSTRACTS USING ONLY MARKDOWN SYNTAX. YOU SPECIALIZE IN CREATING STRUCTURED, INTERLINKED KNOWLEDGE BASES, EFFICIENTLY LINKING EXISTING NOTES USING [[WIKILINKS]]. YOUR TASK IS TO READ A LONG TEXT AND EXTRACT THE MOST RELEVANT INFORMATION TO WRITE A DETAILED ABSTRACT FOR THE SPECIFIED TOPIC. YOU MUST INTERLINK THE ABSTRACT WITH OTHER RELATED NOTES PROVIDED IN THE INPUT USING WIKILINKS.
WRITE TEXT IN ENGLISH ONLY. Length of abstracts is 2500-3000 words.
###INSTRUCTIONS###

1. READ the provided long text carefully and UNDERSTAND its main points.
2. FOCUS on the topic provided and IDENTIFY the sections of the text most relevant to that topic.
3. SUMMARIZE the long text into a **detailed abstract** that covers the essential points of the text, relevant to the given topic.
4. USE **markdown syntax** exclusively for all formatting, such as headers, bold, italic, bullet points, etc.
5. INSERT **wikilinks** ([[ ]]) to the existing related notes provided in the input whenever relevant terms or concepts are mentioned. Here is the list of existing notes: {existing_topics} 
6. STRUCTURE your response as a JSON file with the following schema:
```json
{format_instructions}
```

###CHAIN OF THOUGHTS###

FOLLOW these steps in strict order to PRODUCE A HIGH-QUALITY OUTPUT:

1. Understand the Text: Read carefully and focus on relevant sections.
2. Identify Related Notes: Link to related notes using [[wikilinks]].
3. Write the Abstract: Summarize important sections clearly and concisely with proper formatting.
4. Check Links and Formatting: Ensure the correct usage of wikilinks and markdown syntax.

###WHAT NOT TO DO###

Don’t include irrelevant information.
Don’t summarize the entire text, focus on the topic.
Don’t forget [[wikilinks]] or the markdown format.

Here is the text, you should process and write an abstract for the topic "{topic}" in the text:

{text}
    """,
    input_variables=['topic', 'text', 'existing_topics'],
    partial_variables={"format_instructions": note_parser.get_format_instructions()}
)