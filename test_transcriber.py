import os
import sys
import io
from pydub import AudioSegment
from pytube import YouTube
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from openai import OpenAI


def exit_with_error(message, error):
    print(f"{message}: {error}")
    sys.exit(1)


def cleanup_files(folder_path):
    files_to_delete = ['audio_downloaded_short.mp4', 'audio_downloaded_long.mp4', 'transcript.txt']

    for filename in files_to_delete:
        file_path = os.path.join(folder_path, filename)
        try:
            # Check if the file exists
            if os.path.exists(file_path):
                # Close the file if it's open
                open_files = [f for f in globals() if isinstance(globals()[f], io.IOBase)]
                if file_path in open_files:
                    globals()[file_path].close()
                # Delete the file
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error while deleting file {file_path}: {e}")


def download_audio(video_url, audio_file_path):
    youtube = YouTube(video_url)
    audio = youtube.streams.filter(only_audio=True).first()
    audio.download(filename=audio_file_path)
    return audio


def transcribe_audio(audio_path, transcription_path):
    try:
        audio_file = open(os.path.join(base_path, file_extension), 'rb')
        transcription = client.audio.transcriptions.create(
          model="whisper-1", 
          file=audio_file)
        print('Writing transcript file...')
        with open(transcription_path, "w") as file:
            file.write(transcription.text)
    except Exception as e:
        exit_with_error("Error transcribing audio", e)


def split_audio(audio_path, chunk_size_ms):
    # Load the audio file using PyDub
    print('Grabbing audio from the video...')
    audio = AudioSegment.from_file(audio_path)

    # Calculate number of chunks
    num_chunks = len(audio) // chunk_size_ms + (1 if len(audio) % chunk_size_ms > 0 else 0)
    chunk_paths = []

    # Create directory to store audio chunks if it doesn't exist
    base_path = os.path.splitext(audio_path)[0] + "_chunks"
    os.makedirs(base_path, exist_ok=True)

    # Split the audio and export each chunk
    print('Splitting the audio into discrete files...')
    for i in range(num_chunks):
        start_ms = i * chunk_size_ms
        end_ms = start_ms + chunk_size_ms
        chunk = audio[start_ms:end_ms]
        chunk_file_path = os.path.join(base_path, f"chunk_{i + 1}.mp3")
        chunk.export(chunk_file_path, format="mp3")
        chunk_paths.append(chunk_file_path)

    return chunk_paths


def transcribe_and_append_audio(audio_path, output_file, chunk_size_ms):
    try:
        # Split the audio into chunks
        chunk_paths = split_audio(audio_path, chunk_size_ms)

        # Iterate through each audio chunk and transcribe it
        for chunk_path in chunk_paths:
            print(f'Transcribing audio chunk {chunk_path}...')
            with open(chunk_path, 'rb') as file:
                transcription = client.audio.transcriptions.create(model="whisper-1", file=file)
                # Append the transcribed text to the output file
                with open(output_file, "a") as out_file:
                    out_file.write(transcription.text.strip() + "\n")
    except Exception as e:
        exit_with_error("Error transcribing audio chunks", e)
		

def format_text(input_text, n=100):
    print('Formatting output text...')
    # First pass: Add a newline after each colon
    input_text = input_text.replace(':', ':\n')
    
    # Second pass: Add a newline every n characters, taking the new lines into account
    formatted_text = ''
    current_length = 0  # Track the current length of the line
    
    for word in input_text.split(' '):  # Split the text into words
        word_length = len(word)
        if current_length + word_length > n:
            # If adding the next word exceeds the limit, start a new line
            formatted_text += '\n' + word
            current_length = word_length
        else:
            # Otherwise, add the word to the current line
            if formatted_text:  # Add a space before the word if it's not the start of the text
                formatted_text += ' '
                current_length += 1  # Account for the added space
            formatted_text += word
            current_length += word_length
        
        # Account for newlines within the word itself (e.g., after a colon)
        newline_count = word.count('\n')
        if newline_count > 0:
            # Reset the current length for new lines
            current_length = word_length - word.rfind('\n') - 1
    
    return formatted_text
		


# Load the environment variables and video link
load_dotenv(r"C:\Users\jspri\Desktop\RAG_Model\env_variables.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

YOUTUBE_VIDEO_SHORT = "https://www.youtube.com/watch?v=HQnPp9b7IOk&t=2s"
#YOUTUBE_VIDEO_LONG = "https://www.youtube.com/watch?v=QVKj3LADCnA&list=PLPQAixVXrNTUw464KbIvuvH-86jqHGBlz"
YOUTUBE_VIDEO_LONG = "https://www.youtube.com/watch?v=cdiD-9MMpb0"


# Select the model and load the embeddings
print('Selecting the model...')
MODEL = "gpt-3.5-turbo"
#MODEL = "llama2"
if MODEL.startswith("gpt"):
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
else:
    model = Ollama(model=MODEL)
    embeddings = OllamaEmbeddings(model=MODEL)
	
	
# Define the paths
base_path = r"C:\Users\jspri\Desktop\RAG_Model"

'''
print('Cleaning the folder...')
cleanup_files(base_path)
'''

# Grab the desired audio and put it into a local file
client = OpenAI()

video_length = 'long'

if video_length == 'short':
    video_url = YOUTUBE_VIDEO_SHORT
    file_extension = 'audio_downloaded_short.mp4'
elif video_length == 'long':
    video_url = YOUTUBE_VIDEO_LONG
    file_extension = 'audio_downloaded_long.mp4'

transcription_path = os.path.join(base_path, "transcript.txt")
audio_path = os.path.join(base_path, file_extension)



download_audio(video_url, audio_path)
total_size = (os.path.getsize(audio_path))/1000000
print(f"Total size of the audio file: {total_size:.2f} MB")



input_file = audio_path
chunk_size_ms = 10 * 60 * 1000  # 10 minutes in milliseconds
if total_size < 25:
    print('Downloaded file less than 25 MB. Transcribing audio.')
    transcribe_audio(audio_path, transcription_path)
else: 
    print('Downloaded file greater than 25 MB. Splitting and transcribing audio.')
    transcribe_and_append_audio(audio_path, transcription_path, chunk_size_ms)
	
	
	
try:
    print('Loading text')
    loader = TextLoader(transcription_path)
    text_documents = loader.load()
    if not text_documents:
        raise ValueError("No documents loaded from transcription.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    documents = text_splitter.split_documents(text_documents)
    if not documents:
        raise ValueError("Document splitting failed.")

except Exception as e:
    print(f"An error occurred during text processing: {e}")
    sys.exit(1)
	
	
	
# Set up RAG model
parser = StrOutputParser()
template = """
Based on the context provided, answer the question 
with a detailed explanation. If the question is unclear or 
lacks sufficient context to provide an informed answer, 
respond with "I don't know" or ask for clarification.

Context: {context}

Question: {question}

Please ensure your answer is thorough and detailed, offering 
insights and explanations to support your conclusions.
"""
prompt = PromptTemplate.from_template(template)

# Either build a new datastore or use an existing one
datastore = DocArrayInMemorySearch.from_documents(documents, embeddings)

# Run the RAG model and print the output
chain = (
    {"context": datastore.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

# Run the chain and pass the question
store_data = chain.invoke("Please give a detailed summary of the transcript. Include a breakdown of any pertient equations discussed.")
# Format the text with n=100
formatted_text = format_text(store_data, 100)
# Print the formatted text
print(formatted_text)