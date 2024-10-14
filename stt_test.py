import os
import whisper

# FFmpeg 절대 경로를 환경 변수에 추가
ffmpeg_path = r"C:\ffmpeg\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

# Whisper 모델 로드
model = whisper.load_model("tiny")

# 음성 파일 변환
result = model.transcribe("Track_011.mp3")
print(result["text"])
