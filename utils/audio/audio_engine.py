import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from faster_whisper import WhisperModel

class AudioEngine:
    def __init__(self):
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # self.model_openai = AutoModelForSpeechSeq2Seq.from_pretrained(
        #     "openai/whisper-medium", torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        # )
        
        self.model = WhisperModel("medium.en")
        
        # self.model_openai.to(self.device)

        # self.processor = AutoProcessor.from_pretrained("openai/whisper-medium")
        
        # self.language = "en"
        # self.task = "transcribe"

        # self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.language, task=self.task)
        
        # self.pipe = pipeline(
        #     "automatic-speech-recognition",
        #     model=self.model_openai,
        #     tokenizer=self.processor.tokenizer,
        #     feature_extractor=self.processor.feature_extractor,
        #     max_new_tokens=128,
        #     generate_kwargs={"forced_decoder_ids": self.forced_decoder_ids},
        #     chunk_length_s=30,
        #     batch_size=16,
        #     return_timestamps=True,
        #     torch_dtype=self.torch_dtype,
        #     device="cpu",
        # )

        # self.dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")

    def transcribe(self, audio_path) -> str:
        full_text = ""
        segments, info = self.model.transcribe(audio_path)
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            full_text += segment.text + " "
        
        return full_text

# engine = AudioEngine()
# print(engine.transcribe("announce.mp3"))