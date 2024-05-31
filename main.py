from flask import Flask, request, jsonify
from flask_cors import CORS
from video_engine import VideoSearchEngine

app = Flask(__name__)
CORS(app)
engine = VideoSearchEngine()

@app.route('/search', methods=['POST'])
def search():
    try:
        query = request.json['query']
        visual_results, audio_results = engine.search(query)
        return jsonify({
            'visual_results': visual_results,
            'audio_results': audio_results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
