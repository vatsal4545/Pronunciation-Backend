from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import time
import uuid
import requests
import tempfile
# import soundfile as sf
import numpy as np
import openai
# Try to import and initialize Azure Speech SDK with comprehensive error handling
try:
    import azure.cognitiveservices.speech as speechsdk
    print("=== Azure Speech SDK imported successfully ===")
    
    # Test if Azure SDK can initialize (this is where Error 2153 occurs)
    try:
        # Try to create a minimal config to test initialization
        test_config = speechsdk.SpeechConfig(subscription="test", region="eastus")
        print("=== Azure Speech SDK initialization test passed ===")
        AZURE_SDK_AVAILABLE = True
    except Exception as init_error:
        print(f"=== Azure Speech SDK initialization failed: {init_error} ===")
        print("=== Will use OpenAI TTS as fallback ===")
        AZURE_SDK_AVAILABLE = False
        
except ImportError as import_error:
    print(f"=== Azure Speech SDK import failed: {import_error} ===")
    print("=== Will use OpenAI TTS only ===")
    speechsdk = None
    AZURE_SDK_AVAILABLE = False
except Exception as general_error:
    print(f"=== Azure Speech SDK general error: {general_error} ===")
    print("=== Will use OpenAI TTS only ===")
    speechsdk = None
    AZURE_SDK_AVAILABLE = False

print(f"=== FINAL AZURE_SDK_AVAILABLE: {AZURE_SDK_AVAILABLE} ===")
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import json
import base64
import wave
from flask import send_file

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_SPEECH_ENDPOINT = os.getenv("AZURE_SPEECH_ENDPOINT")

# Session storage (in-memory for demo)
sessions = {}

# Topics and prompts
TOPICS = {
    "Business": "Let's discuss business strategies and professional development.",
    "Technology": "Let's talk about the latest technology trends and innovations.",
    "Interview Roleplay": "I'll be the interviewer. Tell me about your background and skills.",
    "Dating Roleplay": "Let's practice casual conversation as if we're on a first date.",
    "Client Meeting": "Let's roleplay a client meeting. I'll be the client interested in your services."
}

# Initialize the conversation with system prompt
SYSTEM_PROMPT = """
You are an AI pronunciation tutor named Speak Spark. Your job is to engage users in natural conversation while helping them improve their pronunciation and grammar. Follow these guidelines:
1. Keep your responses conversational and engaging but relatively brief (2-3 sentences).
2. After the user speaks, identify any pronunciation errors in their speech.
3. Provide specific, constructive feedback on pronunciation and grammar.
4. Demonstrate correct pronunciation in your response.
5. Encourage users to practice difficult sounds.
6. Stay on the topic the user has selected.
7. Ask open-ended questions to keep the conversation flowing.
8. Correct grammar mistakes naturally within the conversation context.
"""

# @app.route('/api/generate_speech', methods=['POST'])
# def generate_speech_endpoint():
#     data = request.json
#     text = data.get('text', '')
#     session_id = data.get('session_id', '')
    
#     if not text:
#         return jsonify({'error': 'No text provided'}), 400
    
#     try:
#         # Generate speech using Azure TTS
#         speech_file_path = generate_speech(text)
        
#         return jsonify({
#             'audio_url': '/api/get_audio/' + os.path.basename(speech_file_path)
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500



@app.route('/api/generate_speech', methods=['POST'])
def generate_speech_endpoint():
    data = request.json
    text = data.get('text', '')
    session_id = data.get('session_id', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        print(f"Generating speech for: {text[:50]}...")
        print(f"Azure Speech Key: {AZURE_SPEECH_KEY[:10] if AZURE_SPEECH_KEY else 'NOT SET'}...")
        print(f"Azure Speech Region: {AZURE_SPEECH_REGION}")
        
        # Generate speech using Azure TTS
        speech_file_path = generate_speech(text)
        
        return jsonify({
            'audio_url': '/api/get_audio/' + os.path.basename(speech_file_path)
        })
    
    except Exception as e:
        print(f"Speech generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'azure_sdk_available': AZURE_SDK_AVAILABLE,
        'openai_key_set': bool(os.getenv("OPENAI_API_KEY")),
        'azure_key_set': bool(os.getenv("AZURE_SPEECH_KEY")),
        'message': 'Azure disabled, using OpenAI TTS only'
    })
    return jsonify({
        'status': 'healthy',
        'azure_sdk_available': AZURE_SDK_AVAILABLE,
        'openai_key_set': bool(os.getenv("OPENAI_API_KEY")),
        'azure_key_set': bool(os.getenv("AZURE_SPEECH_KEY")),
        'message': 'Azure disabled, using OpenAI TTS only'
    })

@app.route('/api/start_conversation', methods=['POST'])
def start_conversation():
    data = request.json
    topic = data.get('topic', '')
    
    if topic not in TOPICS:
        return jsonify({'error': 'Invalid topic'}), 400
    
    # Create a new session
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        'topic': topic,
        'history': [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": f"Hi there! {TOPICS[topic]} What brings you here today?"}
        ],
        'pronunciation_feedback': [],
        'grammar_feedback': []
    }
    
    return jsonify({
        'session_id': session_id,
        'greeting': f"Hi there! {TOPICS[topic]} What brings you here today?"
    })

@app.route('/api/process_speech', methods=['POST'])
def process_speech():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    session_id = request.form.get('session_id', '')
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    audio_file = request.files['audio']
    
    # Save the audio file temporarily
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        audio_file.save(temp_audio.name)
        temp_audio_path = temp_audio.name
    
    try:
        # Transcribe audio using Whisper
        try:
            # Ensure OpenAI API key is set
            openai.api_key = os.getenv("OPENAI_API_KEY")
            
            with open(temp_audio_path, "rb") as audio_file:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            print(f"Transcription successful: {transcript[:50]}...")
        except Exception as transcription_error:
            print(f"Transcription failed: {transcription_error}")
            return jsonify({'error': f'Audio transcription failed: {str(transcription_error)}'}), 500
        
        # Perform pronunciation assessment (Azure with fallback)
        try:
            pronunciation_assessment = assess_pronunciation_azure(transcript, temp_audio_path)
        except Exception as pronunciation_error:
            print(f"Pronunciation assessment failed: {pronunciation_error}")
            pronunciation_assessment = create_fallback_pronunciation_assessment(transcript)
        
        # Check grammar using OpenAI GPT
        grammar_feedback = check_grammar_openai(transcript)
        
        # Add user message to history
        sessions[session_id]['history'].append({"role": "user", "content": transcript})
        
        # Add feedback to session
        if pronunciation_assessment:
            sessions[session_id]['pronunciation_feedback'].append(pronunciation_assessment)
        if grammar_feedback:
            sessions[session_id]['grammar_feedback'].append(grammar_feedback)
        
        return jsonify({
            'transcript': transcript,
            'pronunciation_assessment': pronunciation_assessment,
            'grammar_feedback': grammar_feedback
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_audio_path)
        except:
            pass

@app.route('/api/get_ai_response', methods=['POST'])
def get_ai_response():
    data = request.json
    session_id = data.get('session_id', '')
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    try:
        # Include pronunciation and grammar feedback in the context
        enhanced_context = create_enhanced_context(session_id)
        
        # Get AI response based on conversation history with feedback
        openai.api_key = os.getenv("OPENAI_API_KEY")
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=enhanced_context,
            max_tokens=200
        )
        
        ai_response = completion.choices[0].message.content
        
        # Add AI response to history
        sessions[session_id]['history'].append({"role": "assistant", "content": ai_response})
        
        # Generate speech using Azure TTS
        speech_file_path = generate_speech(ai_response)
        
        return jsonify({
            'response': ai_response,
            'audio_url': '/api/get_audio/' + os.path.basename(speech_file_path)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_audio/<filename>', methods=['GET'])
def get_audio(filename):
    # Securely join paths to prevent directory traversal attacks
    audio_path = os.path.join(tempfile.gettempdir(), secure_filename(filename))
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype='audio/wav')
    return jsonify({'error': 'Audio file not found'}), 404

@app.route('/api/pronunciation_feedback', methods=['POST'])
def pronunciation_feedback():
    data = request.json
    session_id = data.get('session_id', '')
    word = data.get('word', '')
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    try:
        # Generate example pronunciation for the word
        speech_file_path = generate_speech(word)
        
        # Get detailed pronunciation tips using GPT
        tips = get_detailed_pronunciation_tips(word)
        
        return jsonify({
            'word': word,
            'tips': tips,
            'audio_url': '/api/get_audio/' + os.path.basename(speech_file_path)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def assess_pronunciation_azure(reference_text, audio_path):
    """
    Perform real pronunciation assessment using Azure Speech Services
    """
    # If Azure SDK is not available, return a fallback assessment
    if not AZURE_SDK_AVAILABLE or not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        print("Azure Speech SDK not available - using fallback pronunciation assessment")
        return create_fallback_pronunciation_assessment(reference_text)
    
    try:
        # Configure speech recognition with pronunciation assessment
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY, 
            region=AZURE_SPEECH_REGION
        )
        
        # Configure pronunciation assessment
        pronunciation_config = speechsdk.PronunciationAssessmentConfig(
            reference_text=reference_text,
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Word,
            enable_miscue=True
        )
        
        # Configure audio input
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        
        # Create speech recognizer
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        
        # Apply pronunciation assessment config
        pronunciation_config.apply_to(speech_recognizer)
        
        # Perform recognition with pronunciation assessment
        result = speech_recognizer.recognize_once()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            # Parse pronunciation assessment results
            pronunciation_result = speechsdk.PronunciationAssessmentResult(result)
            
            # Extract detailed results
            assessment_result = {
                'accuracy_score': pronunciation_result.accuracy_score,
                'fluency_score': pronunciation_result.fluency_score,
                'completeness_score': pronunciation_result.completeness_score,
                'pronunciation_score': pronunciation_result.pronunciation_score,
                'word_details': []
            }
            
            # Get word-level details
            words = pronunciation_result.words
            for word in words:
                phonemes = []
                if hasattr(word, 'phonemes') and word.phonemes:
                    phonemes = [{'phoneme': p.phoneme, 'accuracy_score': p.accuracy_score} for p in word.phonemes]
                elif hasattr(word, 'syllables') and word.syllables:
                    for syllable in word.syllables:
                        if hasattr(syllable, 'phonemes') and syllable.phonemes:
                            phonemes.extend([{'phoneme': p.phoneme, 'accuracy_score': p.accuracy_score} for p in syllable.phonemes])
                
                word_detail = {
                    'word': word.word,
                    'accuracy_score': word.accuracy_score,
                    'error_type': str(word.error_type) if word.error_type else 'None',
                    'phonemes': phonemes
                }
                
                assessment_result['word_details'].append(word_detail)
            
            return assessment_result
        
        else:
            return {'error': 'Failed to assess pronunciation', 'reason': result.reason}
    
    except Exception as e:
        print(f"Azure pronunciation assessment error: {str(e)}")
        print("Falling back to simple pronunciation assessment")
        return create_fallback_pronunciation_assessment(reference_text)

# def check_grammar_openai(text):
#     """
#     Check grammar using OpenAI GPT-4o
#     """
#     try:
#         completion = openai.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {
#                     "role": "system", 
#                     "content": """You are a grammar expert. Analyze the given text for grammar errors and provide specific corrections. 
#                     Return a JSON object with:
#                     - 'has_errors': boolean
#                     - 'errors': array of objects with 'original', 'correction', 'explanation'
#                     - 'corrected_text': the fully corrected version
#                     - 'grammar_score': number from 0-100"""
#                 },
#                 {
#                     "role": "user", 
#                     "content": f"Check this text for grammar errors: '{text}'"
#                 }
#             ],
#             response_format={ "type": "json_object" },
#             max_tokens=300
#         )
        
#         grammar_result = json.loads(completion.choices[0].message.content)
#         return grammar_result
    
#     except Exception as e:
#         print(f"Grammar checking error: {str(e)}")
#         return {
#             'has_errors': False,
#             'errors': [],
#             'corrected_text': text,
#             'grammar_score': 85,
#             'error': f'Grammar check failed: {str(e)}'
#         }

def check_grammar_openai(text):
    """
    Check grammar using OpenAI GPT-4o
    """
    try:
        # Ensure OpenAI API key is set
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a grammar expert. Analyze the given text for grammar errors and provide specific corrections. 
                    Return a JSON object with:
                    - 'has_errors': boolean
                    - 'errors': array of objects with 'original', 'correction', 'explanation'
                    - 'corrected_text': the fully corrected version
                    - 'grammar_score': number from 0-100
                    Make sure all strings in the JSON are properly escaped."""
                },
                {
                    "role": "user", 
                    "content": f"Check this text for grammar errors: {json.dumps(text)}"
                }
            ],
            response_format={ "type": "json_object" },
            max_tokens=300
        )
        
        # Use json.loads with error handling
        try:
            grammar_result = json.loads(completion.choices[0].message.content)
        except json.JSONDecodeError as json_error:
            print(f"JSON parsing error: {json_error}")
            # Return fallback result
            return {
                'has_errors': False,
                'errors': [],
                'corrected_text': text,
                'grammar_score': 85,
                'error': f'Grammar check failed: {str(json_error)}'
            }
        
        return grammar_result
    
    except Exception as e:
        print(f"Grammar checking error: {str(e)}")
        return {
            'has_errors': False,
            'errors': [],
            'corrected_text': text,
            'grammar_score': 85,
            'error': f'Grammar check failed: {str(e)}'
        }

def create_enhanced_context(session_id):
    """
    Create enhanced conversation context including pronunciation and grammar feedback
    """
    session = sessions[session_id]
    context = session['history'].copy()
    
    # Add recent pronunciation feedback to context
    if session['pronunciation_feedback']:
        latest_pronunciation = session['pronunciation_feedback'][-1]
        feedback_summary = f"Recent pronunciation assessment: Overall score {latest_pronunciation.get('pronunciation_score', 'N/A')}/100. "
        
        if 'word_details' in latest_pronunciation:
            low_scoring_words = [
                word for word in latest_pronunciation['word_details'] 
                if word.get('accuracy_score', 100) < 70
            ]
            if low_scoring_words:
                feedback_summary += f"Words needing practice: {', '.join([w['word'] for w in low_scoring_words])}. "
        
        context.append({
            "role": "system", 
            "content": f"Pronunciation feedback: {feedback_summary}Gently incorporate this feedback into your response."
        })
    
    # Add recent grammar feedback to context
    if session['grammar_feedback']:
        latest_grammar = session['grammar_feedback'][-1]
        if latest_grammar.get('has_errors', False):
            error_summary = f"Grammar corrections needed: "
            for error in latest_grammar.get('errors', []):
                error_summary += f"'{error.get('original', '')}' -> '{error.get('correction', '')}'. "
            
            context.append({
                "role": "system", 
                "content": f"Grammar feedback: {error_summary}Naturally incorporate these corrections in your response."
            })
    
    return context

# def generate_speech(text):
#     """Generate speech using Azure TTS and return the path to the audio file"""
#     # Configure speech synthesis
#     speech_config = speechsdk.SpeechConfig(
#         subscription=AZURE_SPEECH_KEY, 
#         region=AZURE_SPEECH_REGION
#     )
    
#     # Use a clear, neutral voice for pronunciation teaching
#     speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
    
#     # Create a temporary file for the audio output
#     output_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
    
#     # Configure audio output
#     audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
    
#     # Create a speech synthesizer
#     speech_synthesizer = speechsdk.SpeechSynthesizer(
#         speech_config=speech_config, 
#         audio_config=audio_config
#     )
    
#     # Synthesize text to speech
#     speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
    
#     # Check result
#     if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
#         return output_path
#     else:
#         raise Exception(f"Speech synthesis failed: {speech_synthesis_result.reason}")

def generate_speech(text):
    """Generate speech using Azure TTS with fallback to OpenAI TTS"""
    # If Azure SDK is not available or credentials missing, use OpenAI directly
    if not AZURE_SDK_AVAILABLE or not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        print(f"Using OpenAI TTS (Azure not available): {text[:50]}...")
        return generate_speech_openai(text)
    
    try:
        print(f"Trying Azure TTS for: {text[:50]}...")
        
        # Additional runtime check for Azure platform initialization
        try:
            speech_config = speechsdk.SpeechConfig(
                subscription=AZURE_SPEECH_KEY, 
                region=AZURE_SPEECH_REGION
            )
        except Exception as config_error:
            print(f"Azure config creation failed: {config_error}")
            raise Exception("Azure config failed")
            
        speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
        
        output_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
        
        try:
            audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
        except Exception as audio_config_error:
            print(f"Azure audio config failed: {audio_config_error}")
            raise Exception("Azure audio config failed")
        
        try:
            speech_synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config, 
                audio_config=audio_config
            )
        except Exception as synthesizer_error:
            print(f"Azure synthesizer creation failed: {synthesizer_error}")
            raise Exception("Azure synthesizer failed")
        
        try:
            speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
        except Exception as synthesis_error:
            print(f"Azure synthesis execution failed: {synthesis_error}")
            raise Exception("Azure synthesis execution failed")
        
        if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Azure TTS successful")
            return output_path
        else:
            raise Exception(f"Azure TTS failed with reason: {speech_synthesis_result.reason}")
            
    except Exception as e:
        print(f"Azure TTS failed, falling back to OpenAI: {str(e)}")
        return generate_speech_openai(text)

def generate_speech_openai(text):
    """Fallback speech generation using OpenAI TTS"""
    try:
        print("Using OpenAI TTS...")
        
        # Check if OpenAI API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("OpenAI API key not available")
        
        # Use the simplest OpenAI approach to avoid compatibility issues
        try:
            # Set the API key globally (most compatible approach)
            openai.api_key = api_key
            
            # Create the client without any extra parameters
            response = openai.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=text
            )
            print("OpenAI TTS call successful")
            
        except Exception as openai_error:
            print(f"OpenAI TTS call failed: {openai_error}")
            raise Exception(f"OpenAI TTS failed: {openai_error}")
        
        output_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
        
        # Write the audio content to file
        try:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print("Audio file written successfully")
        except Exception as write_error:
            print(f"File writing failed: {write_error}")
            raise Exception("Failed to write audio file")
        
        print("OpenAI TTS successful")
        return output_path
        
    except Exception as e:
        print(f"OpenAI TTS error: {str(e)}")
        raise Exception(f"Speech generation failed: {str(e)}")


def create_fallback_pronunciation_assessment(reference_text):
    """
    Create a fallback pronunciation assessment when Azure is not available
    """
    words = reference_text.split()
    word_details = []
    
    for word in words:
        word_detail = {
            'word': word,
            'accuracy_score': 85,  # Default score
            'error_type': 'None',
            'phonemes': []
        }
        word_details.append(word_detail)
    
    return {
        'accuracy_score': 85,
        'fluency_score': 80,
        'completeness_score': 90,
        'pronunciation_score': 85,
        'word_details': word_details,
        'note': 'Fallback assessment - Azure Speech SDK not available'
    }

def get_detailed_pronunciation_tips(word):
    """Get detailed pronunciation tips for a specific word using GPT"""
    try:
        # Ensure OpenAI API key is set
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a pronunciation expert. Provide detailed, practical tips for pronouncing English words correctly. Include phonetic breakdown, common mistakes, and practice suggestions."
                },
                {
                    "role": "user", 
                    "content": f"Provide detailed pronunciation tips for the word '{word}'"
                }
            ],
            max_tokens=200
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"Focus on saying '{word}' clearly. Break it down syllable by syllable and practice slowly before increasing speed."

if __name__ == '__main__':
    # Ensure temp directory exists
    os.makedirs(os.path.join(tempfile.gettempdir()), exist_ok=True)
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
