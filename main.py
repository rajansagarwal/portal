from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.search.search_engine import SearchEngine

print("STARTING UP")
app = Flask(__name__)
print("INITIALIZING CORS")
CORS(app)
print("INITIALIZING SEARCH ENGINIE")
engine = SearchEngine()

print("THIS IS THE COLLECION")
print(engine.collection.get())

@app.route('/search', methods=['POST', 'GET'])
def search():
    try:
        query = request.json['query']
        print(f"Querying {query}")
        results = engine.query(query, "text")
        return jsonify({
            'results': results
        })
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
