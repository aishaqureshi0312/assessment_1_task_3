from flask import Flask, jsonify, request, render_template
import pandas as pd
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)
CORS(app)

# Load the dataset
df = pd.read_csv("/Users/aishaqureshi/Desktop/Test/song_dataset.csv")

# Create a user-song matrix
user_song_matrix = df.pivot_table(index='user', columns='song', values='play_count', fill_value=0)

# Normalize the play_count values
scaler = MinMaxScaler()
normalized_matrix = pd.DataFrame(scaler.fit_transform(user_song_matrix), columns=user_song_matrix.columns, index=user_song_matrix.index)

# Calculate the similarity between users
user_similarity = cosine_similarity(normalized_matrix)

def recommend_song(user_id):
    try:
        # Get the user's index from the user_song_matrix
        user_index = user_song_matrix.index.get_loc(user_id)

        # Get the user's row from the similarity matrix
        user_row = user_similarity[user_index]

        # Find the user most similar to the given user
        similar_user_index = user_row.argmax()

        # Get the songs the similar user has listened to
        similar_user_songs = user_song_matrix.loc[user_song_matrix.index[similar_user_index]]

        # Get the songs the given user has already listened to
        user_songs = user_song_matrix.loc[user_id]

        # Find songs that the similar user has listened to but the given user has not
        recommended_songs = similar_user_songs[user_songs == 0]

        # Get the top recommended song
        top_song = recommended_songs.idxmax()

        return top_song

    except IndexError:
        # Handle the case where the user_id is not found in the user_song_matrix
        raise BadRequest("User ID not found")

# Endpoint to render the HTML page
@app.route('/')
def index():
    return render_template('logic1.html')

# API endpoint to get user IDs
@app.route('/get_user_ids', methods=['GET', 'POST'])
def get_user_ids():
    user_ids = user_song_matrix.index.tolist()
    return jsonify({"user_ids": user_ids})

# API endpoint to get the top unheard songs for a user
@app.route('/get_top_unheard_songs', methods=['POST'])
def get_top_unheard_songs():
    try:
        data = request.get_json(force=True)
        user_id = data.get('user_id')

        print("Received request at /get_top_unheard_songs with user_id:", user_id)

        if user_id is None:
            raise BadRequest("User ID parameter is missing")

        # Get the user's index from the user_song_matrix
        user_index = user_song_matrix.index.get_loc(user_id)

        # Get the songs the given user has already listened to
        user_songs = user_song_matrix.loc[user_id]

        # Find songs that the given user has not listened to
        unheard_songs = user_songs[user_songs == 0].index.tolist()

        # Sort unheard songs based on play count in descending order
        unheard_songs = sorted(unheard_songs, key=lambda x: df.loc[df['song'] == x, 'play_count'].values[0], reverse=True)

        # Select the top 10 unheard songs based on play count
        top_unheard_songs = unheard_songs[:10]

        print("Top Unheard Songs:", top_unheard_songs)

        return jsonify({"user_id": user_id, "top_unheard_songs": top_unheard_songs})

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        print("Exception in get_top_unheard_songs:", str(e))
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)