from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os
import cv2
from moviepy.editor import VideoFileClip
import tempfile

from utils.audio.audio_engine import AudioEngine
from utils.video.photo_engine import PhotoEngine
from utils.embeddings.embeddings_engine import EmbeddingsEngine
from utils.clustering.clustering_engine import ClusteringEngine
from utils.summarization.summary_engine import SummaryEngine
from utils.search.search_engine import SearchEngine
from PIL import Image

class VideoSearchEngine:
    def __init__(self):
        print("Initializing Audio Engine")
        self.audio_engine = AudioEngine()
        print("Initializing Photo Engine")
        self.photo_engine = PhotoEngine("default")
        print("Initializing Embeddings Engine")
        self.embeddings_engine = EmbeddingsEngine("default")
        print("Initializing Summary Engine")
        self.summary_engine = SummaryEngine()
        print("Initializing Chroma Search Engine")
        self.search_engine = SearchEngine()

        self.video_fragments_dir = "store"
        os.makedirs(self.video_fragments_dir, exist_ok=True)
        self.interval = 60
        self.csv_filename = "extracted.csv"
        self.load_existing_videos()

    def load_existing_videos(self):
        if os.path.exists(self.csv_filename):
            df = pd.read_csv(self.csv_filename)
            video_ids = df['Video ID'].tolist()
            self.existing_videos = {video_id: 1 for video_id in video_ids}
        else:
            self.existing_videos = {}

    def extract_frames(self, video_path, interval, fps):
        print("Extracting Frames")
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        seconds_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count == int(seconds_count * fps):
                frames.append((seconds_count, frame))
                seconds_count += interval
            frame_count += 1
        cap.release()
        return frames

    def process_video(self, video_path):
        video_clip = VideoFileClip(video_path)
        fps = video_clip.fps
        frames = self.extract_frames(video_path, self.interval, fps)

        for seconds, frame in frames:
            video_id = f"{video_path}::{seconds}"
            if self.search_engine.exists_in_collection(video_id):
                print(f"Skipping already indexed frame: {video_id}")
                continue

            description = self.photo_engine.describe_image(frame)
            start_time = seconds
            duration = self.interval
            end_time = min(start_time + duration, video_clip.duration)

            if end_time > start_time:
                video_fragment = video_clip.subclip(start_time, end_time)
                with tempfile.NamedTemporaryFile(delete=True, suffix='.wav') as temp_audio_file:
                    video_fragment.audio.write_audiofile(temp_audio_file.name)
                    transcription = self.audio_engine.transcribe(temp_audio_file.name)
                    summary = self.summary_engine.summarize(transcription)

                concatenated_description = f"{description} {summary}"
                self.search_engine.add(concatenated_description, seconds, video_path)

    def process_image(self, image_path):
        with Image.open(image_path) as img:
            description = self.photo_engine.describe_image(img)
            self.search_engine.add(description, 0, image_path)

    def process_all_files(self, directory_path):
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.endswith(".mp4"):
                    self.process_video(file_path)
                elif file_path.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.process_image(file_path)
                else:
                    print(f"Skipping unsupported file type: {file_path}")

    def search(self, query_text):
        results = self.search_engine.query(query_text, "text")
        return results