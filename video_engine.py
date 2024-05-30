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
from utils.search.search_engine import SearchEngine
from utils.summarization.summary_engine import SummaryEngine
class VideoSearchEngine:
    def __init__(self):
        self.audio_engine = AudioEngine()
        print("Audio engine initialized.")
        self.photo_engine = PhotoEngine("default")
        print("Photo engine initialized.")
        self.embeddings_engine = EmbeddingsEngine("default")
        print("Embeddings engine initialized.")
        self.search_engine = SearchEngine(data=[])
        print("Search engine initialized.")
        self.summary_engine = SummaryEngine()
        print("Summary engine initialized.")
        self.stored_data = {}
        print("Stored data structure initialized.")

        self.video_fragments_dir = "store"
        self.interval = 1800
        self.csv_filename = "extracted.csv"
        self.existing_videos = set()

        if os.path.exists(self.csv_filename):
            existing_df = pd.read_csv(self.csv_filename)
            self.existing_videos.update(existing_df['Video Filename'].dropna().unique())
            print("Loaded existing video filenames.")

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return []
        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % self.interval == 0:
                frames.append((frame_count, frame))
                print(f"Extracted frame {frame_count}")
            frame_count += 1
        cap.release()
        print("Finished extracting frames.")
        return frames

    def compress_video(self, input_path, output_path, crf=28):
        command = f"ffmpeg -i {input_path} -vcodec libx264 -crf {crf} {output_path}"
        os.system(command)
        print(f"Compressed video saved: {output_path}")
        os.remove(input_path)
        print(f"Original video fragment removed: {input_path}")

    def process_video(self, video_path):
        video_id = os.path.basename(video_path)

        if any(video_id in filename for filename in self.existing_videos):
            print(f"Skipping {video_id}, already processed.")
            return

        video_clip = VideoFileClip(video_path)
        fps = video_clip.fps

        frames = self.extract_frames(video_path)
        frame_descriptions = []
        frame_indices = []
        audio_transcriptions = []
        summaries = []
        video_filenames = []

        for frame_number, frame in frames:
            description = self.photo_engine.describe_image(frame)
            frame_descriptions.append(description)
            frame_indices.append(frame_number)

            start_time = frame_number / fps
            duration = self.interval / fps
            end_time = min(start_time + duration, video_clip.duration)

            if end_time > start_time:
                video_fragment = video_clip.subclip(start_time, end_time)
                video_filename = os.path.join(self.video_fragments_dir, f"{video_id}_fragment_{frame_number}.mp4")
                compressed_filename = os.path.join(self.video_fragments_dir, f"{video_id}_fragment_{frame_number}_compressed.mp4")

                video_fragment.write_videofile(video_filename, codec="libx264", fps=video_clip.fps)
                print(f"Video fragment saved: {video_filename}")

                self.compress_video(video_filename, compressed_filename)
                video_filenames.append(compressed_filename)

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_audio_file:
                    video_fragment.audio.write_audiofile(temp_audio_file.name)
                    try:
                        transcription = self.audio_engine.transcribe(temp_audio_file.name)
                        summary = self.summary_engine.summarize(transcription)
                        audio_transcriptions.append(transcription.summary_text)
                        summaries.append(summary)
                    except Exception as e:
                        print(f"Error during transcription: {str(e)}")
                        audio_transcriptions.append("Error in transcription")
                        summaries.append("Error in summary")
            else:
                print(f"Skipped zero-length video for frame {frame_number}")
                audio_transcriptions.append("")
                summaries.append("")
                video_filenames.append("")

        self.stored_data[video_id] = {
            'frame_indices': frame_indices,
            'frame_descriptions': frame_descriptions,
            'audio_transcriptions': audio_transcriptions,
            'video_filenames': video_filenames,
            'Summary': summaries
        }

        self.save_to_csv(video_id, frame_indices, frame_descriptions, audio_transcriptions, video_filenames, summaries)

    def save_to_csv(self, video_id, frame_indices, frame_descriptions, audio_transcriptions, video_filenames, summaries):
        df = pd.DataFrame({
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

    def process_all_videos(self, directory_path):
        for filename in os.listdir(directory_path):
            if filename.endswith(".mp4"):
                video_path = os.path.join(directory_path, filename)
                print(f"Processing video: {video_path}")
                self.process_video(video_path)
            else:
                print(f"Skipping non-video file: {filename}")

video_search = VideoSearchEngine()
video_search.process_all_videos('videos')