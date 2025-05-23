import openai
import os

# Ensure the OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

client = openai.OpenAI()

def transcribe_audio(audio_file_path):
    """
    Transcribes audio to text using OpenAI Whisper.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        str: The transcribed text.
    """
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None 