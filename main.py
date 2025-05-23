from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

import tempfile # Added for temporary file handling

# Import your service modules
from stt_service import transcribe_audio
from tts_service import synthesize_speech

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# API Keys - These are loaded by the service modules themselves now.
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
# AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY not found in .env file")
# if not AZURE_SPEECH_KEY:
#     raise ValueError("AZURE_SPEECH_KEY not found in .env file")
# if not AZURE_SPEECH_REGION:
#     raise ValueError("AZURE_SPEECH_REGION not found in .env file. Please add it (e.g., AZURE_SPEECH_REGION=westus)")

@app.route('/')
def hello():
    return "Pronunciation Tutor Backend is running!"

@app.route('/api/transcribe', methods=['POST'])
def transcribe_endpoint():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']

    # Save the audio file temporarily because Whisper SDK expects a file path
    # You might want to implement a more robust solution for handling file uploads
    # (e.g., saving to a specific directory, unique naming, cleaning up old files)
    temp_dir = tempfile.mkdtemp()
    temp_audio_path = os.path.join(temp_dir, audio_file.filename)
    audio_file.save(temp_audio_path)

    try:
        transcript_text = transcribe_audio(temp_audio_path)
        if transcript_text is not None:
            return jsonify({"transcription": transcript_text})
        else:
            return jsonify({"error": "Failed to transcribe audio"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the temporary file and directory
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        os.rmdir(temp_dir)

@app.route('/api/synthesize-speech', methods=['POST'])
def synthesize_speech_endpoint():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text_input = data['text']

    try:
        audio_bytes, mime_type = synthesize_speech(text_input)
        
        if audio_bytes and mime_type:
            return Response(audio_bytes, mimetype=mime_type)
        else:
            return jsonify({"error": "Failed to synthesize speech"}), 500
    except Exception as e:
        # Log the exception e for more detailed server-side debugging
        print(f"Error during speech synthesis endpoint: {str(e)}") 
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True) 