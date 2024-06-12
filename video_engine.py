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

    def compress_video(self, input_path, output_path, crf=31, overwrite=False):
        if not overwrite and os.path.exists(output_path):
            print(f"Skipping compression for {output_path} as it already exists.")
            return
        
        command = f"ffmpeg -i {input_path} -vcodec libx264 -crf {crf} {output_path}"
        os.system(command)
        print(f"Compressed and saved video to {output_path}.")

    def process_video(self, video_path):
        print(self.existing_videos)
        video_id = os.path.basename(video_path)
        compressed_video_path = os.path.join("videos", video_id.replace(".mp4", "_compressed.mp4"))
        self.compress_video(video_path, compressed_video_path)

        video_clip = VideoFileClip(compressed_video_path)
        fps = video_clip.fps
        frames = self.extract_frames(compressed_video_path, self.interval, fps)

        for seconds, frame in frames:
            print(f"PROCESSING {video_path} FRAME at {seconds} seconds")
            description = self.photo_engine.describe_image(frame)
            start_time = seconds
            duration = self.interval
            end_time = min(start_time + duration, video_clip.duration)

            if end_time > start_time:
                fragment_filename = os.path.join(self.video_fragments_dir, f"{os.path.basename(video_path)}_{seconds}.mp4")
                video_fragment = video_clip.subclip(start_time, end_time)
                video_fragment.write_videofile(fragment_filename, codec="libx264")

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_audio_file:
                    video_fragment.audio.write_audiofile(temp_audio_file.name)
                    transcription = self.audio_engine.transcribe(temp_audio_file.name)
                    summary = self.summary_engine.summarize(transcription)

                concatenated_description = f"{description} {summary}"
                # embedding = self.embeddings_engine.embed(concatenated_description)
                self.search_engine.add(concatenated_description, description, transcription, summary, fragment_filename)
                self.save_to_csv(compressed_video_path, [seconds], [description], [transcription], [fragment_filename], [summary[0]['summary_text']])

    def process_all_videos(self, directory_path):
        for filename in os.listdir(directory_path):
            if filename.endswith(".mp4"):
                video_path = os.path.join(directory_path, filename)
                if f"""videos/{filename.replace(".mp4", "_compressed.mp4")}""" not in self.existing_videos:
                    self.process_video(video_path)
                else:
                    print(f"Skipping already processed video: {video_path}")
                
    def save_to_csv(self, video_id, frame_indices, frame_descriptions, audio_transcriptions, video_filenames, summaries):
        df = pd.DataFrame({
            'Video ID': [video_id] * len(frame_indices),
            'Frame Index': frame_indices,
            'Frame Description': frame_descriptions,
            'Audio Transcription': audio_transcriptions,
            'Video Filename': video_filenames,
            'Summary': summaries
        })
        if os.path.exists(self.csv_filename):
            df.to_csv(self.csv_filename, mode='a', header=False, index=False)
        else:
            df.to_csv(self.csv_filename, index=False)
        print(f"Data appended to CSV for {video_id}.")

    def search(self, query_text):
        results = self.search_engine.query(query_text, "text")
        return results

engine = VideoSearchEngine()
engine.process_all_videos("input")
results = engine.search("knowledge graphs")
print(results)