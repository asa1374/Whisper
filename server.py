import os
import time
import whisper
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
    config={'app_name': "Whisper API"}
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# FFmpeg 절대 경로를 환경 변수에 추가
ffmpeg_path = r"C:\ffmpeg\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

# Whisper 모델 로드
model = whisper.load_model("tiny")
# model = whisper.load_model("small")
# model = whisper.load_model("medium")

# 오디오 파일을 텍스트로 변환하는 엔드포인트
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # 파일 확장자 확인
    allowed_extensions = {'wav', 'mp3'}
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return jsonify({"error": "wav와 mp3만 지원합니다."}), 400

    # 파일 저장 경로
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    print(f"Saved file to: {filepath}")

    # Whisper 모델을 사용하여 음성을 텍스트로 변환
    try:
        start_time = time.time()  # 시작 시간 기록
        result = model.transcribe(filepath)
        end_time = time.time()  # 종료 시간 기록
        execution_time = end_time - start_time  # 실행 시간 계산
    except RuntimeError as e:
        return jsonify({"error": f"Failed to transcribe audio: {str(e)}"}), 500

    text = result['text']

    # 파일 삭제
    os.remove(filepath)
    print(f"Execution time: {execution_time:.2f} seconds")  # 실행 시간 출력

    return jsonify({
        "transcription": text,
        "execution_time": f"{execution_time:.2f} seconds"
    })


# 기본 라우트
@app.route('/')
def home():
    return "STT test of OpenAI whisper"


if __name__ == '__main__':
    # uploads 디렉토리 생성
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(host='0.0.0.0', port=5000)
