import os
import torch
from yt_dlp import YoutubeDL
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline
from dotenv import load_dotenv
from pydub import AudioSegment
import openai   

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# 1. Download YouTube audio
def download_audio(youtube_url, output_path="audio"):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav"},
        ],
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path

# 2. Load diarization pipeline
def load_diarizer():
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    return pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 3. Load Whisper ASR pipeline
def load_asr():
    return hf_pipeline("automatic-speech-recognition", model="openai/whisper-large-v2")

# 4. Summarize text using GPT-4
def summarize(text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Summarize this person's arguments or opinions:\n\n{text}"}],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()

# 5. Process full video
def process_video(youtube_url):
    audio_path = download_audio(youtube_url)  # Returns 'audio'
    diarizer = load_diarizer()
    diarization = diarizer(audio_path + ".wav")  # Uses 'audio.wav'
    asr = load_asr()

    segments = list(diarization.itertracks(yield_label=True))
    speaker_texts = {}

    # Load full audio once
    audio = AudioSegment.from_wav(audio_path + ".wav")

    for segment, _, speaker in segments:
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        chunk = audio[start_ms:end_ms]

        chunk_path = f"tmp_chunk_{speaker}_{int(start_ms)}.wav"
        chunk.export(chunk_path, format="wav")

        print(f"[{speaker}] From {segment.start:.2f}s to {segment.end:.2f}s")

        result = asr(chunk_path, return_timestamps=True)
        text = result["text"]
        speaker_texts.setdefault(speaker, []).append(text)

        os.remove(chunk_path)

    # Summarize each speaker
    for speaker, utterances in speaker_texts.items():
        full_text = " ".join(utterances)
        summary = summarize(full_text)
        print(f"\n==== {speaker} ====" + "\n" + summary + "\n")

# 6. Entry point
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python summarize_youtube.py <youtube_url>")
        exit(1)
    process_video(sys.argv[1])
