import os
import tempfile
from flask import Flask, jsonify, request
import whisper

app = Flask(__name__)

model = whisper.load_model("base")

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    _, temp_path = tempfile.mkstemp()
    file.save(temp_path)

    audio = whisper.load_audio(temp_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    result = model.transcribe(mel)

    os.remove(temp_path)

    return jsonify({'transcription': result['text']})

if __name__ == '__main__':
    app.run(debug=True)
