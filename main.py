from flask import Flask, request, jsonify
from flask_cors import CORS
from video_engine import VideoSearchEngine

app = Flask(__name__)
CORS(app)
engine = VideoSearchEngine()
engine.load_csv_data()

@app.route('/search', methods=['POST', 'GET'])
def search():
    try:
        query = request.json['query']
        user = request.json['user']
        visual_results, audio_results, summary_results, video_ids, video_filepaths = engine.search(query, user)
        return jsonify({
            'visual_results': visual_results,
            'audio_results': audio_results,
            'summary_results': summary_results,
            'video_filepaths': video_filepaths
        })
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
