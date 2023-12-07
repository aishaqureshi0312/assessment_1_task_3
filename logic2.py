from flask import Flask, jsonify, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS
from werkzeug.exceptions import BadRequest  # Import the BadRequest class

app = Flask(__name__)
CORS(app)

# Load the dataset
df = pd.read_csv("/Users/aishaqureshi/Desktop/Assessment_1_task_3/song_dataset.csv")

# Create a user-song matrix
user_song_matrix = df.pivot_table(index='user', columns='song', values='play_count', fill_value=0)

# Normalize the play_count values
scaler = MinMaxScaler()
normalized_matrix = pd.DataFrame(scaler.fit_transform(user_song_matrix), columns=user_song_matrix.columns, index=user_song_matrix.index)


# Calculate the similarity between users
user_similarity = cosine_similarity(normalized_matrix)



def recommend_song(user_id, selected_song):
    # Get the user's index from the user_song_matrix
    user_index = user_song_matrix.index.get_loc(user_id)

    # Get the user's row from the similarity matrix
    user_row = user_similarity[user_index]

    # Find the user most similar to the given user
    similar_user_index = user_row.argsort()[-10:][::-1]  # Get the indices of the top 10 most similar users
    similar_user_songs = user_song_matrix.iloc[similar_user_index].sum()  # Sum of songs listened by similar users

    # Get the songs the given user has already listened to
    user_songs = user_song_matrix.loc[user_id]

    # Find songs that the similar users have listened to but the given user has not
    recommended_songs = similar_user_songs[user_songs == 0]

    # Check if the selected song is in the recommended songs, if yes, remove it
    if selected_song in recommended_songs:
        recommended_songs = recommended_songs[recommended_songs.index != selected_song]

    # If there are no recommended songs based on user similarity, fallback to top songs
    if recommended_songs.empty:
        recommended_songs = user_song_matrix.sum().sort_values(ascending=False).head(10)

    # If there are still no recommended songs, return an empty list
    if recommended_songs.empty:
        return []

    # Get the top recommended songs (up to 10) along with their titles
    top_songs_info = recommended_songs.sort_values(ascending=False).head(10)
    top_songs = [{"song_id": song_id, "title": df[df['song'] == song_id]['title'].values[0]} for song_id in top_songs_info.index]

    return top_songs



# Endpoint to render the HTML page
@app.route('/')
def index():
    return render_template('recommendation.html')

# API endpoint to get user IDs
@app.route('/get_user_ids', methods=['GET', 'POST'])
def get_user_ids():
    user_ids = user_song_matrix.index.tolist()
    return jsonify({"user_ids": user_ids})

# API endpoint to get songs listened by a user
@app.route('/get_user_songs', methods=['POST'])
def get_user_songs():
    try:
        data = request.get_json(force=True)  # Parse JSON even if content type is not set to application/json
        print("Received data:", data)  # Add this line for debugging

        if data is None:
            raise BadRequest("Invalid JSON payload")

        user_id = data.get('user_id')
        if user_id is None:
            raise BadRequest("User ID parameter is missing")

        user_songs = user_song_matrix.loc[user_id]
        listened_songs = [{"song_id": song_id, "title": df[df['song'] == song_id]['title'].values[0]} for song_id in user_songs[user_songs > 0].index]

        return jsonify({"user_id": user_id, "listened_songs": listened_songs})

    except BadRequest as e:
        print("Bad Request in get_user_songs:", str(e))
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        print("Exception in get_user_songs:", str(e))
        import traceback
        traceback.print_exc()  # Print the traceback
        return jsonify({"error": "Internal server error"}), 500

    
@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    try:
        data = request.get_json()

        user_id = data.get('user_id')
        selected_song = data.get('selected_song')

        if user_id is None or selected_song is None:
            return jsonify({"error": "User ID or selected song parameter is missing"}), 400

        recommended_song = recommend_song(user_id, selected_song)

        return jsonify({"recommended_song": recommended_song})

    except Exception as e:
        print("Exception in get_recommendation:", str(e))
        return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)


