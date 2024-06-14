import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from faster_whisper import WhisperModel

class AudioEngine:
    def __init__(self):
        self.model = WhisperModel("medium.en")
        
    def transcribe(self, audio_path) -> str:
        full_text = ""
        segments, info = self.model.transcribe(audio_path)
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            full_text += segment.text + " "
        
        return full_text
