from flask import Flask, jsonify, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
# Load the dataset
df = pd.read_csv("/Users/aishaqureshi/Desktop/Test/song_dataset.csv")
# Display the column names
print(df.columns)

# Create a user-song matrix
user_song_matrix = df.pivot_table(index='user', columns='song', values='play_count', fill_value=0)

# Normalize the play_count values
scaler = MinMaxScaler()
normalized_matrix = pd.DataFrame(scaler.fit_transform(user_song_matrix), columns=user_song_matrix.columns, index=user_song_matrix.index)

# Calculate the similarity between users
user_similarity = cosine_similarity(normalized_matrix)

def recommend_song(user_id):
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



# Endpoint to get song titles based on user ID
@app.route('/get_song_titles', methods=['GET'])
def get_song_titles():
    user_id = request.args.get('user_id')

    # Try function
    if user_id is None:
        return jsonify({"error": "User ID parameter is missing"}), 400

    # Find the corresponding SongRecord for the given user ID
    # user_record = next((record for record in song_records if record.user == user_id), None)
    recommended_song = recommend_song(user_id)
    return recommended_song



if __name__ == '__main__':
    app.run(debug=True, port=5001)
