import os
import time
import whisper
from faster_whisper import WhisperModel
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint


# Flask 애플리케이션 초기화
app = Flask(__name__)

# Swagger 설정
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Whisper & Faster Whisper API"}
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# FFmpeg 절대 경로를 환경 변수에 추가
ffmpeg_path = r"C:\ffmpeg\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

# CUDA 및 모델 설정
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# whisper-v3-turbo 모델 로드 (OpenAI Whisper)
whisper_model_id = "openai/whisper-large-v3-turbo"
whisper_v3_turbo_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    whisper_model_id, torch_dtype=torch_dtype
)
whisper_v3_turbo_model.to(device)
processor = AutoProcessor.from_pretrained(whisper_model_id)

# 파이프라인 설정
pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model_id,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# 파일 확장자 유효성 확인
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 오디오 파일 로드
def load_audio(filepath):
    waveform, sampling_rate = librosa.load(filepath, sr=None)
    return waveform, sampling_rate

# Whisper (OpenAI) 모델을 사용하여 음성을 텍스트로 변환하는 엔드포인트
@app.route('/transcribe/whisper', methods=['POST'])
def transcribe_with_whisper():
    if 'file' not in request.files or 'model_size' not in request.form:
        return jsonify({"error": "No file or model size provided"}), 400

    file = request.files['file']
    model_size = request.form['model_size']
    if not allowed_file(file.filename):
        return jsonify({"error": "wav와 mp3만 지원합니다."}), 400

    # Whisper 모델 로드 (OpenAI Whisper)
    try:
        whisper_model = whisper.load_model(model_size)
    except RuntimeError as e:
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

    # 파일 저장
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)
    print(f"Saved file to: {filepath}")

    try:
        start_time = time.time()
        result = whisper_model.transcribe(filepath)
        end_time = time.time()
        execution_time = end_time - start_time
    except RuntimeError as e:
        return jsonify({"error": f"Failed to transcribe audio: {str(e)}"}), 500

    # 파일 삭제
    os.remove(filepath)

    return jsonify({
        "transcription": result['text'],
        "execution_time": f"{execution_time:.2f} seconds"
    })

# Faster Whisper 모델을 사용하여 음성을 텍스트로 변환하는 엔드포인트
@app.route('/transcribe/faster-whisper', methods=['POST'])
def transcribe_with_faster_whisper():
    if 'file' not in request.files or 'model_size' not in request.form:
        return jsonify({"error": "No file or model size provided"}), 400

    file = request.files['file']
    model_size = request.form['model_size']
    if not allowed_file(file.filename):
        return jsonify({"error": "wav와 mp3만 지원합니다."}), 400

    # Faster Whisper 모델 로드
    try:
        faster_whisper_model = WhisperModel(f"{model_size}", device="cpu", compute_type="int8")
    except RuntimeError as e:
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

    # 파일 저장
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)
    print(f"Saved file to: {filepath}")

    try:
        start_time = time.time()
        segments, info = faster_whisper_model.transcribe(filepath)
        text = ' '.join([segment.text for segment in segments])
        end_time = time.time()
        execution_time = end_time - start_time
    except RuntimeError as e:
        return jsonify({"error": f"Failed to transcribe audio: {str(e)}"}), 500

    # 파일 삭제
    os.remove(filepath)

    return jsonify({
        "transcription": text,
        "detected_language": info.language,
        "execution_time": f"{execution_time:.2f} seconds"
    })

# whisper-v3-turbo 모델을 사용하여 음성을 텍스트로 변환하는 엔드포인트
@app.route('/transcribe/whisper-v3-turbo', methods=['POST'])
def transcribe_with_whisper_v3_turbo():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({"error": "wav와 mp3만 지원합니다."}), 400

    # 파일 저장
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)
    print(f"Saved file to: {filepath}")

    try:
        start_time = time.time()

        # librosa로 파일 로드
        waveform, sampling_rate = load_audio(filepath)

        # 파이프라인을 사용하여 텍스트 변환
        audio = {"raw": waveform, "sampling_rate": sampling_rate}
        result = pipe(audio, return_timestamps=True)
        text = result["text"]

        end_time = time.time()
        execution_time = end_time - start_time
    except RuntimeError as e:
        return jsonify({"error": f"Failed to transcribe audio: {str(e)}"}), 500

    # 파일 삭제
    os.remove(filepath)

    return jsonify({
        "transcription": text,
        "execution_time": f"{execution_time:.2f} seconds"
    })

# 기본 라우트
@app.route('/')
def home():
    return "STT test of Whisper and Faster Whisper"

# uploads 디렉토리 생성
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5000)
