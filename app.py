from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd  # Import pandas for DataFrame handling
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the movie datasets and KNN model
movies = pd.read_csv(r"E:\project\movies.csv")  # Use raw string to handle backslashes
final_dataset = pd.read_csv(r"E:\project\final_dataset.csv")  # Use raw string for the path
with open(r"E:\project\knn.pkl", 'rb') as file:  # Use raw string for loading the KNN model
    knn = pickle.load(file)


from scipy.sparse import load_npz

# Load the sparse matrix from NPZ file
csr_data = load_npz('csr_data.npz')

# Ensure csr_data is defined (you should load or define this variable)
# csr_data = ...  # Load or prepare your CSR data for KNN

def get_recommendation(movie_name):
    # Perform a case-insensitive search for the movie title
    movie_list = movies[movies['title'].str.contains(movie_name, case=False, na=False)]
    if len(movie_list) > 0:
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        
        # Get recommendations from the KNN model
        distance, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=11)
        rec_movies_indices = sorted(list(zip(indices.squeeze().tolist(), distance.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        
        # Prepare the list of recommended movies
        recommended_movies = []
        for val in rec_movies_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommended_movies.append({'Title': movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
        
        return recommended_movies  # Return the list of recommendations
    else:
        return "Movie not found..."

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')  # Ensure this matches your HTML file's location

# Route to handle the search request
@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()  # Get the JSON data sent from the client
    query = data.get('query', '').strip()  # Extract and clean the query

    try:
        recommendations = get_recommendation(query)  # Get recommendations based on user input
        if isinstance(recommendations, str):  # Check if the response is an error message
            return jsonify({'error': recommendations}), 404  # Return a 404 error if not found
        
        return jsonify({'result': recommendations})  # Return the list of recommendations
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Handle any errors during processing

if __name__ == '__main__':
    app.run(debug=True)  # Start the Flask app in debug mode
