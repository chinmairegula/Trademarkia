import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the USPTO ID manual dataset
with open('idmanual.json', 'r') as file:
    dataset = json.load(file)

# Preprocess the dataset
descriptions = [entry['id_tx'] for entry in dataset]
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(descriptions)

# Build the recommendation model
model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
model.fit(X)

@app.route('/algorithm', methods=['GET', 'POST'])
def recommend_class():
    if request.method == 'POST':
        data = request.json
        user_input = data['goods_and_services']
        user_vector = vectorizer.transform([user_input])

        # Find the nearest neighbors
        _, indices = model.kneighbors(user_vector)

        # Get the recommended classes
        recommended_classes = [dataset[idx]['Class Name'] for idx in indices[0]]

        return jsonify({'recommended_classes': recommended_classes})
    else:
        return jsonify({
        "id_tx": "90909933-23",
        "class_id": "129",
        "description": "bank details of the servers",
        "status": "b"
    },
            {
                "id_tx": "001-2345",
                "class_id": "444",
                "description": "medical records management system",
                "status": "P"
            },
            {
                "id_tx": "678-9012",
                "class_id": "777",
                "description": "supply chain logistics tracking software",
                "status": "A"
            },
            {
                "id_tx": "333-4444",
                "class_id": "222",
                "description": "customer relationship management tool",
                "status": "C"
            },
    {
        "id_tx": "343-2323",
        "class_id": "345",
        "description": "cse servers and the banking algorithms",
        "status": "M"
    })

if __name__ == '__main__':
    app.run()
