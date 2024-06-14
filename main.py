from flask import Flask, request, jsonify
from flask_cors import CORS
from video_engine import VideoSearchEngine

print("STARTING UP")
app = Flask(__name__)
print("INITIALIZING CORS")
CORS(app)
print("INITIALIZING VIDEO ENGINIE")
engine = VideoSearchEngine()
engine.search_engine.delete_collection("portal_db")
engine.search_engine.create_collection("portal_db")
engine.process_all_files("input")

print(engine.search_engine.collection.get())

@app.route('/search', methods=['POST', 'GET'])
def search():
    try:
        query = request.json['query']
        print(f"Querying {query}")
        results = engine.search(query)
        return jsonify({
            'results': results
        })
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
