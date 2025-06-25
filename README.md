## YouTube Speaker Diarizer & Summarizer (06-17-2025)

Paste in a YouTube link, run one command, and get speaker-by-speaker summaries of what was said. This tool is built to break down long-form videos like podcasts, debates, and interviews so you can quickly understand who said what without scrubbing through the whole thing.

---

### What It Does

- Detects and separates individual speakers
- Transcribes each person's audio segment
- Summarizes each speaker's points into clean paragraphs
- Outputs everything directly to your terminal

---

### Tech Stack Overview

| Component       | Tool/Library                     | Purpose                                |
|----------------|----------------------------------|----------------------------------------|
| Video Download | yt-dlp                           | Downloads YouTube audio as .wav        |
| Diarization     | pyannote.audio                   | Distinguishes between speakers         |
| Transcription   | Whisper (openai/whisper-large-v2) | Converts audio to text                 |
| Summarization   | OpenAI GPT-4                     | Generates speaker-specific summaries   |
| Audio Slicing   | pydub                            | Extracts speaker-specific audio chunks |
| Env Management  | python-dotenv                    | Loads API keys from `.env`             |
| Runtime         | torch                            | Enables GPU/CPU inference switching    |

---

### Folder Structure

```
youtube_diarizer_06_17_2025/
├── summarize_youtube.py       # Main script
├── audio.wav                  # Output audio file
├── audio/                     # Temp files (auto-cleaned)
├── .env                       # API keys (not committed)
└── /venv (yt_diarizer)        # Conda environment
```

---

### .env Format

Create a `.env` file in your project root with:

```
HF_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your_openai_key_here
```

---

### Setup Instructions

```bash
# Create environment
conda create -n yt_diarizer python=3.10 -y
conda activate yt_diarizer

# Install dependencies
pip install yt-dlp ffmpeg-python python-dotenv torch torchaudio pyannote.audio
pip install scipy openai transformers pydub

# Run the pipeline
python summarize_youtube.py "https://www.youtube.com/watch?v=your_video_id"
```

---

### Why This Project Stands Out

Most tools either dump a rough transcript or summarize the video as one big block of text. This one does more. It tells you who said what, breaks everything down by speaker, and gives you a clear picture of the conversation. If you're analyzing interviews, podcasts, or panel discussions, this saves hours of manual work.

