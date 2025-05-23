import azure.cognitiveservices.speech as speechsdk
import os
# import base64 # No longer needed for direct byte return

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

if not AZURE_SPEECH_KEY:
    raise ValueError("AZURE_SPEECH_KEY not found in .env file")
if not AZURE_SPEECH_REGION:
    raise ValueError("AZURE_SPEECH_REGION not found in .env file")

def synthesize_speech(text_input):
    """
    Synthesizes speech from text using Azure TTS.

    Args:
        text_input (str): The text to synthesize.

    Returns:
        tuple: (audio_bytes, mime_type_string) or (None, None) if synthesis fails.
               audio_bytes are the raw audio data (WAV format).
               mime_type_string is 'audio/wav'.
    """
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)
    # Using a None audio_config will result in a PullAudioOutputStream, which can be read for the audio data in-memory.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    result = speech_synthesizer.speak_text_async(text_input).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        audio_data = result.audio_data
        return audio_data, "audio/wav"  # Return raw bytes and MIME type
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print(f"Error details: {cancellation_details.error_details}")
        return None, None
    return None, None 