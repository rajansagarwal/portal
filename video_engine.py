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
        self.stored_data = {}
        print("Stored data structure initialized.")
        
        self.video_fragments_dir = "videos"

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
            if frame_count % 600 == 0:  # Extract every 600th frame
                frames.append((frame_count, frame))
                print(f"Extracted frame {frame_count}")
            frame_count += 1
        cap.release()
        print("Finished extracting frames.")
        return frames

    def process_video(self, video_path):
        video_id = os.path.basename(video_path)
        video_clip = VideoFileClip(video_path)
        fps = video_clip.fps
        
        frames = self.extract_frames(video_path)
        frame_descriptions = []
        frame_indices = []
        audio_transcriptions = []
        video_filenames = []

        # Process each frame to extract and save video fragments
        for frame_number, frame in frames:
            description = self.photo_engine.describe_image(frame)
            frame_descriptions.append(description)
            frame_indices.append(frame_number)

            start_time = frame_number / fps
            duration = 600 / fps  # duration of each video fragment corresponding to frames
            end_time = min(start_time + duration, video_clip.duration)

            if end_time > start_time:  # Ensuring the video segment is non-zero in duration
                video_fragment = video_clip.subclip(start_time, end_time)
                video_filename = os.path.join(self.video_fragments_dir, f"{video_id}_fragment_{frame_number}.mp4")
                video_fragment.write_videofile(video_filename, codec="libx264", fps=video_clip.fps)
                print(f"Video fragment saved: {video_filename}")
                video_filenames.append(video_filename)

                # Temporarily extract audio for transcription without saving it
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_audio_file:
                    video_fragment.audio.write_audiofile(temp_audio_file.name)
                    try:
                        transcription = self.audio_engine.transcribe(temp_audio_file.name)
                        audio_transcriptions.append(transcription)
                    except Exception as e:
                        print(f"Error during transcription: {str(e)}")
                        audio_transcriptions.append("Error in transcription")
            else:
                print(f"Skipped zero-length video for frame {frame_number}")
                audio_transcriptions.append("")
                video_filenames.append("")

        self.stored_data[video_id] = {
            'frame_indices': frame_indices,
            'frame_descriptions': frame_descriptions,
            'audio_transcriptions': audio_transcriptions,
            'video_filenames': video_filenames
        }

        self.save_to_csv(video_id, frame_indices, frame_descriptions, audio_transcriptions, video_filenames)

    def save_to_csv(self, video_id, frame_indices, frame_descriptions, audio_transcriptions, video_filenames):
        df = pd.DataFrame({
            'Frame Index': frame_indices,
            'Frame Description': frame_descriptions,
            'Audio Transcription': audio_transcriptions,
            'Video Filename': video_filenames
        })
        df.to_csv(f'{video_id}_descriptions.csv', index=False)
        print(f"Data saved to CSV for {video_id}.")

    def process_all_videos(self, directory_path):
        for filename in os.listdir(directory_path):
            if filename.endswith(".mp4"):
                video_path = os.path.join(directory_path, filename)
                print(f"Processing video: {video_path}")
                self.process_video(video_path)
            else:
                print(f"Skipping non-video file: {filename}")

# Example usage
video_search = VideoSearchEngine()
video_search.process_video('connexsci-live.mp4')
