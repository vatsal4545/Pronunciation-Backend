
# Speak Spark Tutor Backend

This is the backend for the Speak Spark Tutor, an AI-powered pronunciation feedback system. It provides real-time audio processing, speech transcription, pronunciation assessment, and AI-driven conversation.

## Features

- Real-time speech-to-text using OpenAI Whisper
- Pronunciation error detection and feedback
- AI conversation generation with GPT-4o
- Text-to-speech using Azure Speech Services
- Session management for continued conversations

## Setup Instructions

1. Clone this repository to your local machine
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   AZURE_SPEECH_KEY=your_azure_speech_key_here
   AZURE_SPEECH_ENDPOINT=https://your_region.api.cognitive.microsoft.com/
   AZURE_SPEECH_REGION=your_region_here
   ```
4. Run the Flask application:
   ```
   python app.py
   ```
   
The server will start on http://localhost:5000

## API Endpoints

### Start Conversation
- **URL**: `/api/start_conversation`
- **Method**: `POST`
- **Body**:
  ```json
  {
    "topic": "Business"
  }
  ```
- **Response**:
  ```json
  {
    "session_id": "uuid-string",
    "greeting": "Hi there! Let's discuss business strategies and professional development. What brings you here today?"
  }
  ```

### Process Speech
- **URL**: `/api/process_speech`
- **Method**: `POST`
- **Body**: Form data with:
  - `audio`: Audio file (WAV format)
  - `session_id`: Session ID string
- **Response**:
  ```json
  {
    "transcript": "The user's transcribed speech",
    "pronunciation_issues": [
      {
        "word": "example",
        "problematic_sound": "l",
        "confidence": 0.7,
        "correction_tip": "Focus on the 'l' sound in 'example'."
      }
    ]
  }
  ```

### Get AI Response
- **URL**: `/api/get_ai_response`
- **Method**: `POST`
- **Body**:
  ```json
  {
    "session_id": "uuid-string"
  }
  ```
- **Response**:
  ```json
  {
    "response": "The AI tutor's response",
    "audio_url": "/api/get_audio/filename.wav"
  }
  ```

### Get Pronunciation Feedback
- **URL**: `/api/pronunciation_feedback`
- **Method**: `POST`
- **Body**:
  ```json
  {
    "session_id": "uuid-string",
    "word": "difficult"
  }
  ```
- **Response**:
  ```json
  {
    "word": "difficult",
    "tips": "Tips for pronouncing this word",
    "audio_url": "/api/get_audio/filename.wav"
  }
  ```

## Frontend Integration

Connect your React frontend to these API endpoints. Here's an example of how to call the API from your frontend:

```javascript
// Example: Starting a conversation
const startConversation = async (topic) => {
  try {
    const response = await fetch('http://localhost:5000/api/start_conversation', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ topic }),
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error starting conversation:', error);
    throw error;
  }
};

// Example: Sending audio for processing
const processAudio = async (audioBlob, sessionId) => {
  try {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
    formData.append('session_id', sessionId);
    
    const response = await fetch('http://localhost:5000/api/process_speech', {
      method: 'POST',
      body: formData,
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error processing audio:', error);
    throw error;
  }
};
```

## Further Customization

The backend can be extended with:
- More sophisticated pronunciation assessment algorithms
- Additional language support
- User account management
- Long-term progress tracking
- Custom voice selection for TTS
