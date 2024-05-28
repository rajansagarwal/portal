import numpy as np
import os
import cv2
from tqdm import tqdm

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
        self.stored_data = {}  # Initialize the stored data dictionary
        print("Stored data structure initialized.")

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
            if frame_count > 1000:
                break
            if frame_count % 100 == 0:
                frames.append(frame)
                print(f"Extracted frame {frame_count}")
            frame_count += 1
        cap.release()
        print("Finished extracting frames.")
        return frames

    def process_video(self, video_path):
        video_id = os.path.basename(video_path)
        
        frames = self.extract_frames(video_path)
        frame_descriptions = []
        for frame in tqdm(frames, desc="Describing frames"):
            description = self.photo_engine.describe_image(frame)
            frame_descriptions.append(description)

        frame_embeddings = [self.embeddings_engine.embed(description) for description in tqdm(frame_descriptions, desc="Creating embeddings")]

        audio_transcription = self.audio_engine.transcribe(video_path)
        transcription_embedding = self.embeddings_engine.embed(audio_transcription)

        self.stored_data[video_id] = {
            'frame_descriptions': frame_descriptions,
            'frame_embeddings': frame_embeddings,
            'transcription': audio_transcription,
            'transcription_embedding': transcription_embedding
        }

        all_text_data = frame_descriptions + [audio_transcription]
        
        print(all_text_data)
        
        print("=======")
        print(self.stored_data)
        # self.search_engine.add_data(all_text_data)

        # self.index_video(video_path, frame_embeddings, transcription_embedding)

    def index_video(self, video_path, frame_embeddings, transcription_embedding):
        combined_embedding = np.mean([np.array(frame_embeddings), transcription_embedding], axis=0)
        self.search_engine.add_data([combined_embedding])
        print(f"Combined embedding added for {video_path}.")

    def search_videos(self, query):
        query_embedding = self.embeddings_engine.embed(query)
        print(f"Query embedding created for: '{query}'")
        nearest_neighbours = self.search_engine.query(query_embedding)
        print("Query processed. Retrieving results.")
        return self.search_engine.query_text_results(nearest_neighbours)

    def process_all_videos(self, directory_path):
        for filename in os.listdir(directory_path):
            if filename.endswith(".mp4"):
                video_path = os.path.join(directory_path, filename)
                print(f"Processing video: {video_path}")
                self.process_video(video_path)
            else:
                print(f"Skipping non-video file: {filename}")

video_search = VideoSearchEngine()
video_search.process_video('connexsci-live.mp4')
results = video_search.search_videos("Knowledge graphs")
print(results)
