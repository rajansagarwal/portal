import numpy as np
import os
import cv2
from tqdm import tqdm
import pandas as pd
from moviepy.editor import AudioFileClip, VideoFileClip
import tempfile

from utils.audio.audio_engine import AudioEngine
from utils.video.photo_engine import PhotoEngine
from utils.embeddings.embeddings_engine import EmbeddingsEngine
from utils.clustering.clustering_engine import ClusteringEngine
from utils.search.search_engine import SearchEngine
from utils.summarization.summary_engine import SummaryEngine

class VideoSearchEngine:
    def __init__(self):
        print("Initializing Audio Engine")
        self.audio_engine = AudioEngine()
        print("Initializing Photo Engine")
        self.photo_engine = PhotoEngine("default")
        print("Initializing Embeddings Engine")
        self.embeddings_engine = EmbeddingsEngine("default")
        print("Initializing Search Engine")
        self.search_engine = SearchEngine(data=[])
        print("Initializing Summary Engine")
        self.summary_engine = SummaryEngine()
        print("Initializing Clustering Engine")
        self.classifier = ClusteringEngine(embeddings_engine=self.embeddings_engine, threshold=0.75)

        self.video_fragments_dir = "store"
        os.makedirs(self.video_fragments_dir, exist_ok=True)
        self.interval = 60
        self.csv_filename = "extracted.csv"
        self.existing_videos = set()

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

    def compress_video(self, input_path, output_path, crf=31):
        command = f"ffmpeg -i {input_path} -vcodec libx264 -crf {crf} {output_path}"
        os.system(command)

    def process_video(self, video_path):
        compressed_video_path = video_path.replace(".mp4", "_compressed.mp4")
        self.compress_video(video_path, compressed_video_path)

        video_clip = VideoFileClip(compressed_video_path)
        fps = video_clip.fps

        frames = self.extract_frames(compressed_video_path, self.interval, fps)

        frame_indices = []
        frame_descriptions = []
        audio_transcriptions = []
        video_filenames = []
        summaries = []

        for seconds, frame in frames:
            print(f"PROCESSING FRAME at {seconds} seconds")
            description = self.photo_engine.describe_image(frame)

            start_time = seconds
            duration = self.interval
            end_time = min(start_time + duration, video_clip.duration)

            if end_time > start_time:
                fragment_filename = os.path.join(self.video_fragments_dir, f"fragment_{seconds}.mp4")
                video_fragment = video_clip.subclip(start_time, end_time)
                video_fragment.write_videofile(fragment_filename, codec="libx264")

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_audio_file:
                    print("Video Fragment")
                    video_fragment.audio.write_audiofile(temp_audio_file.name)
                    print("Transcription")
                    transcription = self.audio_engine.transcribe(temp_audio_file.name)
                    print("Summary")
                    summary = self.summary_engine.summarize(transcription)
                    print("Cluster Add")
                    combined_description = f"{description}. {summary[0]['summary_text']}"
                    self.classifier.add_event_description(combined_description)

            frame_indices.append(seconds)
            frame_descriptions.append(description)
            audio_transcriptions.append(transcription)
            video_filenames.append(fragment_filename)
            summaries.append(summary[0]['summary_text'])

        self.save_to_csv(compressed_video_path, frame_indices, frame_descriptions, audio_transcriptions, video_filenames, summaries)

    def process_all_videos(self, directory_path):
        for filename in os.listdir(directory_path):
            if filename.endswith(".mp4"):
                video_path = os.path.join(directory_path, filename)
                self.process_video(video_path)

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

video_search = VideoSearchEngine()
video_search.process_all_videos('videos')