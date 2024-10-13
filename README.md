# Whisper & Faster Whisper API

이 프로젝트는 Whisper, Faster Whisper, Whisper v3 Turbo 모델을 사용하여 음성 파일을 텍스트로 변환하는 Flask 기반의 API 서버입니다.

## 기능
- Whisper(OpenAI)를 사용한 음성 텍스트 변환
- Faster Whisper 모델을 사용한 음성 텍스트 변환
- Whisper v3 Turbo 모델을 사용한 음성 텍스트 변환
- Swagger UI를 통해 API 테스트 가능

## 요구 사항
- Python 3.8+
- PyTorch
- Flask
- ffmpeg

## 설치 방법

1. 이 저장소를 클론합니다:
   ```bash
   git clone https://github.com/사용자이름/whisper-api.git
   cd whisper-api
   
2. 필요한 라이브러리를 설치합니다:
   ```bash
   pip install -r requirements.txt
   
3. FFmpeg을 설치하고 환경 변수에 FFmpeg 경로를 추가합니다.

4. uploads 디렉토리를 생성합니다:
   ```bash
   mkdir uploads


## 실행 방법

1. Flask 애플리케이션을 실행합니다:
   ```bash
   python app.py

2. 브라우저에서 Swagger UI에 접속하여 API를 테스트할 수 있습니다:
   ```bash
   http://localhost:5000/swagger

## API 엔드포인트

### 1. Whisper(OpenAI)를 사용한 음성 텍스트 변환
- **POST** `/transcribe/whisper`
- **파라미터**:
- `file`: 음성 파일 (`wav` 또는 `mp3` 형식)
- `model_size`: 모델 크기 (`tiny`, `small`, `medium`, `large`)
- **응답**:
- `transcription`: 변환된 텍스트
- `execution_time`: 변환에 걸린 시간

### 2. Faster Whisper를 사용한 음성 텍스트 변환
- **POST** `/transcribe/faster-whisper`
- **파라미터**:
- `file`: 음성 파일 (`wav` 또는 `mp3` 형식)
- `model_size`: 모델 크기 (`tiny`, `small`, `medium`, `large`, `deepdml/faster-whisper-large-v3-turbo-ct2`)
- **응답**:
- `transcription`: 변환된 텍스트
- `detected_language`: 감지된 언어
- `execution_time`: 변환에 걸린 시간

### 3. Whisper v3 Turbo를 사용한 음성 텍스트 변환
- **POST** `/transcribe/whisper-v3-turbo`
- **파라미터**:
- `file`: 음성 파일 (`wav` 또는 `mp3` 형식)
- **응답**:
- `transcription`: 변환된 텍스트
- `execution_time`: 변환에 걸린 시간


