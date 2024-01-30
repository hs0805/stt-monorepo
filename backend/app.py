from flask import Flask, request, jsonify
import deepspeech
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the relative path to the DeepSpeech model files
DEEPSPEECH_MODEL = os.path.join(current_directory, 'deepspeech-0.9.3-models')

# Initialize DeepSpeech model
english_model = deepspeech.Model(DEEPSPEECH_MODEL + '.pbmm')
english_model.enableExternalScorer(DEEPSPEECH_MODEL + '.scorer')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    file_path = None
    try:
        # Check if the POST request has the file part
        if 'audioFile' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        audio_file = request.files['audioFile']

        # If user does not select file, browser also submits an empty part without filename
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # If the file is present and is a WAV file
        if audio_file and audio_file.filename.endswith('.wav'):
            # Save the file to the server
            file_path = os.path.join(current_directory, 'uploaded.wav')
            audio_file.save(file_path)

            # Transcribe the audio
            text = transcribe_audio_file(file_path)

            return jsonify({'text': text}), 200
        else:
            return jsonify({'error': 'Invalid file format'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Remove the file if it exists
        if file_path:
            try:
                os.remove(file_path)
            except Exception as e:
                pass

def transcribe_audio_file(audio_file):
    # Read the audio data from the file
    with open(audio_file, 'rb') as f:
        audio_data = f.read()

    # Convert the audio data to a numpy array of int16 values
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    # Transcribe the audio
    text = english_model.stt(audio_np)

    return text

if __name__ == '__main__':
    app.run(host='localhost', port=4000)
