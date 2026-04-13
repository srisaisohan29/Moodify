import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask import session
import os
from dotenv import load_dotenv
import random
import traceback
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# Load environment variables
load_dotenv()

# Spotify API credentials
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
REDIRECT_URI = 'http://127.0.0.1:5000/callback'
SCOPE = 'streaming user-read-email user-read-private user-modify-playback-state user-read-playback-state'

def create_spotify_oauth():
    return SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE
    )

def get_token():
    """Get the token for the current session"""
    token_info = session.get('token_info', None)
    if not token_info:
        raise Exception("No token found in session")
    return token_info['access_token']

def get_spotify():
    """Get an authenticated Spotify client"""
    auth_manager = create_spotify_oauth()
    if not session.get('token_info', False):
        raise Exception("No token info in session")
    return spotipy.Spotify(auth_manager=auth_manager)

# Playlist IDs for different emotions
PLAYLIST_IDS = {
    'Happy': '3fOFxhMsiIHmen3UHwCQLE',
    'Sad': '0qfJuUyAOpZ9n4UXO9BvIf',
    'Angry': '3UW3eURYZ96cgylRtgCKlj',
    'Surprised': '0XC3JqBvCvTfAV9RFdB5wD',
    'Fearful': '5fl2kEaHKbsEU5gxqXvYRY',
    'Disgusted': '1Klo8yQr6uEjMmnRyWqLqe',
    'Neutral': '0GRBMC2r2cg6x8rnvJwEMQ'
}

# Emotion-to-query mapping for dynamic recommendations with expanded keywords
EMOTION_KEYWORDS = {
    "Happy": "upbeat energetic happy vibe chill-mood",
    "Sad": "melancholic-emotional-sad-romantic yesudas heart-break failure depressed alone",
    "Angry": "intense-aggressive-powerful angry Telangana-movement-songs angry-young-man introduction-songs mass main-character-energy mafia",
    "Surprised": "party-dance honey-singh latest-Telangana-private-dj-songs disco Bollywood-party-hits all-time-Bollywood-non-stop-mix",
    "Fearful": "ghost-scary horror-creepy loud-scream ghost-sounds horror-mix",
    "Disgusted": "dark-intense electronic-devil dark-funk-phonk trending-insta-funk phonk",
    "Neutral": "allu-arjun anirudh Prabhas Mahesh-babu NTR ram-charan ranbir-kapoor shahrukh-khan-hits popular-trending-latest-hits telugu-pop"
}

# Language-specific keywords for search queries
LANGUAGE_KEYWORDS = {
    'telugu': ["telugu", "tollywood"],
    'hindi': ["hindi", "bollywood"],
    'english': ["english", "taylor-swift", "dua-lipa", "billie-eilish", "ed-sheeran", "ariana-grande", "harry-styles", "bruno-mars", "michael-jackson"],
    'instrumental': ["instrumental", "bgm", "ar-rahman-instrumental", "hans-zimmer"],
    'classic': ["classic", "hits", "1980-2010", "90s", "80s", "Chiranjeevi", "nusrat-fateh-ali-khan", "Kishore-kumar", "bappi-lahiri", "old"]
}

# Fallback query keywords
FALLBACK_KEYWORDS = "popular trending songs all time hits"

def get_playlist_tracks(emotion, limit=10):
    """Get tracks from a curated playlist based on emotion"""
    try:
        sp = get_spotify()
        playlist_id = PLAYLIST_IDS.get(emotion, PLAYLIST_IDS['Neutral'])
        results = sp.playlist_tracks(playlist_id, limit=limit)
        
        return _process_tracks(results['items'])
    except Exception as e:
        print(f"Error getting playlist tracks: {str(e)}")
        return []

def get_dynamic_recommendations(emotion, limit=40):
    """Get dynamically recommended tracks based on emotion with language distribution using playlists"""
    try:
        sp = get_spotify()
        print(f"Processing recommendation for emotion: {emotion}")
        
        # Get emotion keywords as individual search terms
        emotion_keywords_list = EMOTION_KEYWORDS.get(emotion, EMOTION_KEYWORDS['Neutral']).split()
        
        # Add the emotion itself as a keyword
        all_emotion_keywords = [emotion] + emotion_keywords_list
        
        # Randomly select 5 keywords from the emotion keywords (or all if less than 5)
        if len(all_emotion_keywords) > 5:
            selected_keywords = random.sample(all_emotion_keywords, 5)
        else:
            selected_keywords = all_emotion_keywords
        
        print(f"Selected keywords for {emotion}: {', '.join(selected_keywords)}")
        
        # Target distribution for song selection
        targets = {
            'telugu': 20,   # 20 Telugu songs
            'hindi': 11,    # 11 Hindi songs
            'english': 5,   # 5 English songs
            'instrumental': 2,  # 2 Instrumental songs
            'classic': 2    # 2 Classic songs
        }
        
        # Track IDs to avoid duplicates
        track_ids = set()
        all_songs_by_type = {
            'telugu': [],
            'hindi': [],
            'english': [],
            'instrumental': [],
            'classic': []
        }
        
        # Function to find top 5 playlists for a given type using multiple keyword searches
        def get_top_playlists_for_type(type_name):
            playlists_found = {}
            
            # For all types, search with each keyword
            for keyword in selected_keywords:
                # Build query combining emotion keyword and language
                if type_name in LANGUAGE_KEYWORDS:
                    # Get language keywords list
                    lang_keywords = LANGUAGE_KEYWORDS[type_name]
                    
                    # Select one random language keyword
                    if lang_keywords:
                        # Randomly select one language keyword
                        lang_keyword = random.choice(lang_keywords)
                        individual_search_query = f"{keyword} {lang_keyword}"
                        
                        try:
                            # Get multiple playlists for each keyword (top 5) then randomly select one
                            playlist_results = sp.search(q=individual_search_query, type='playlist', limit=5, market='IN')
                            if not playlist_results or 'playlists' not in playlist_results or 'items' not in playlist_results['playlists'] or not playlist_results['playlists']['items']:
                                print(f"No playlists found for query: {individual_search_query}")
                                continue
                            
                            # Filter out any None values
                            valid_playlists = [p for p in playlist_results['playlists']['items'] 
                                              if p is not None and isinstance(p, dict) and 'id' in p]
                            
                            if not valid_playlists:
                                print(f"No valid playlists found for query: {individual_search_query}")
                                continue
                            
                            # Randomly select one playlist from the results
                            playlist = random.choice(valid_playlists)
                            
                            # Add this playlist if it's not already found
                            if playlist['id'] not in playlists_found:
                                playlists_found[playlist['id']] = playlist
                                print(f"Found random playlist for keyword: {keyword} {lang_keyword}")
                            
                        except Exception as e:
                            print(f"Error searching for playlists with query {individual_search_query}: {str(e)}")
                            continue
                else:
                    # For types without specific language keywords
                    search_query = f"{keyword} {type_name}"
                    
                    try:
                        # Get multiple playlists for each keyword (top 5) then randomly select one
                        playlist_results = sp.search(q=search_query, type='playlist', limit=5, market='IN')
                        if not playlist_results or 'playlists' not in playlist_results or 'items' not in playlist_results['playlists'] or not playlist_results['playlists']['items']:
                            print(f"No playlists found for query: {search_query}")
                            continue
                        
                        # Filter out any None values
                        valid_playlists = [p for p in playlist_results['playlists']['items'] 
                                          if p is not None and isinstance(p, dict) and 'id' in p]
                        
                        if not valid_playlists:
                            print(f"No valid playlists found for query: {search_query}")
                            continue
                        
                        # Randomly select one playlist from the results
                        playlist = random.choice(valid_playlists)
                        
                        # Add this playlist if it's not already found
                        if playlist['id'] not in playlists_found:
                            playlists_found[playlist['id']] = playlist
                            print(f"Found random playlist for keyword: {keyword} {type_name}")
                        
                    except Exception as e:
                        print(f"Error searching for playlists with query {search_query}: {str(e)}")
                        continue
            
            print(f"Total unique playlists found for {type_name}: {len(playlists_found)}")
            
            # Filter out any None values just to be safe
            result_playlists = list(playlists_found.values())
            result_playlists = [p for p in result_playlists if p is not None and isinstance(p, dict) and 'id' in p]
            return result_playlists
        
        # Function to get tracks from a list of playlists
        def get_tracks_from_playlists(playlists, max_per_playlist=20):
            all_tracks = []
            
            for playlist in playlists:
                try:
                    # Skip None playlists
                    if playlist is None or not isinstance(playlist, dict) or 'id' not in playlist:
                        print("Skipping invalid playlist (no id found)")
                        continue
                        
                    # Get tracks from this playlist
                    playlist_tracks_result = sp.playlist_tracks(
                        playlist['id'], 
                        fields='items(track(name,id,artists,album,duration_ms,preview_url,external_urls,popularity))',
                        limit=50
                    )
                    
                    if not playlist_tracks_result or 'items' not in playlist_tracks_result:
                        print(f"No tracks found in playlist {playlist['id']}")
                        continue
                    
                    # Process each track in the playlist
                    playlist_tracks = []
                    for item in playlist_tracks_result['items']:
                        # Skip None or empty items
                        if not item:
                            continue
                            
                        track = item.get('track')
                        # Skip None or empty tracks 
                        if not track or not track.get('id'):
                            continue
                            
                        # Skip tracks already collected
                        if track['id'] in track_ids:
                            continue
                            
                        # Check if album info exists
                        if not track.get('album') or not track['album'].get('images') or not track['album']['images']:
                            # Use default image if album image is missing
                            album_image = '/static/start-vinyl.png'
                        else:
                            album_image = track['album']['images'][0]['url']
                            
                        # Process the track
                        processed_track = {
                            'name': track['name'],
                            'artist': track['artists'][0]['name'] if track['artists'] else 'Unknown',
                            'url': track['external_urls']['spotify'] if 'external_urls' in track else '',
                            'id': track['id'],
                            'preview_url': track.get('preview_url', ''),
                            'duration_ms': track.get('duration_ms', 0),
                            'album_image': album_image,
                            'popularity': track.get('popularity', 0)
                        }
                        
                        playlist_tracks.append(processed_track)
                    
                    # Balance popularity with random selection for more variety
                    # Sort by popularity
                    playlist_tracks.sort(key=lambda x: x.get('popularity', 0), reverse=True)
                    
                    # Select tracks using a hybrid approach:
                    # - 80% from top popular tracks
                    # - 20% randomly selected from the rest
                    selected_tracks = []
                    
                    if playlist_tracks:
                        # Calculate how many tracks to take from popular vs random selections
                        popular_count = min(int(max_per_playlist * 0.8), len(playlist_tracks))
                        random_count = min(max_per_playlist - popular_count, len(playlist_tracks) - popular_count)
                        
                        # Get top popular tracks
                        selected_tracks.extend(playlist_tracks[:popular_count])
                        
                        # Get random tracks from the remaining pool
                        if random_count > 0 and len(playlist_tracks) > popular_count:
                            random_pool = playlist_tracks[popular_count:]
                            try:
                                random_selections = random.sample(random_pool, random_count)
                                selected_tracks.extend(random_selections)
                            except ValueError:
                                # If not enough tracks for sampling, just take what's available
                                selected_tracks.extend(random_pool)
                        
                        # Add selected tracks to all_tracks and track_ids
                        for track in selected_tracks:
                            track_ids.add(track['id'])
                            all_tracks.append(track)
                    
                except Exception as e:
                    try:
                        playlist_id = playlist['id'] if playlist and isinstance(playlist, dict) and 'id' in playlist else "unknown"
                        print(f"Error getting tracks from playlist {playlist_id}: {str(e)}")
                    except:
                        print(f"Error getting tracks from invalid playlist: {str(e)}")
                    continue
            
            return all_tracks
        
        # Process each type
        for type_name in all_songs_by_type.keys():
            print(f"Processing {type_name} type...")
            
            # Get top 5 playlists for this type
            top_playlists = get_top_playlists_for_type(type_name)
            print(f"Found {len(top_playlists)} playlists for {type_name}")
            
            # Get popular tracks from these playlists
            if top_playlists:
                type_tracks = get_tracks_from_playlists(top_playlists)
                all_songs_by_type[type_name] = type_tracks
                print(f"Got {len(type_tracks)} unique {type_name} tracks")
            else:
                print(f"No playlists found for {type_name}")
                all_songs_by_type[type_name] = []
        
        # Randomly select the required number of tracks from each type
        final_tracks = []
        remaining_types = []
        total_target = sum(targets.values())  # Should be 40
        
        # First pass - take what we have available up to the target for each type
        for type_name, target_count in targets.items():
            available = len(all_songs_by_type[type_name])
            
            if available >= target_count:
                # Randomly select the target count
                selected = random.sample(all_songs_by_type[type_name], target_count)
                final_tracks.extend(selected)
                # Remove selected tracks from the pool
                for track in selected:
                    all_songs_by_type[type_name].remove(track)
            else:
                # Take all available tracks
                final_tracks.extend(all_songs_by_type[type_name])
                # Record the shortage
                shortage = target_count - available
                print(f"Shortage of {shortage} tracks for {type_name}")
                # Add this type to the list for compensation
                remaining_types.append((type_name, shortage))
        
        # Compensate for shortages by taking more tracks from other types
        if remaining_types:
            # Sort remaining types by priority (those with more available tracks first)
            compensation_priority = ['hindi', 'english', 'instrumental', 'telugu', 'classic']
            
            for type_name, shortage in remaining_types:
                # Check if we already have 40 tracks
                if len(final_tracks) >= total_target:
                    print(f"Already have {len(final_tracks)} tracks, stopping compensation")
                    break
                    
                # Calculate remaining slots for compensation
                remaining_slots = total_target - len(final_tracks)
                
                # Adjust shortage if necessary
                actual_shortage = min(shortage, remaining_slots)
                
                # Try to compensate from each type in priority order
                for comp_type in compensation_priority:
                    # Check if we already have 40 tracks
                    if len(final_tracks) >= total_target:
                        break
                        
                    # Skip the type that has the shortage
                    if comp_type == type_name:
                        continue
                    
                    available_comp = len(all_songs_by_type[comp_type])
                    if available_comp > 0:
                        # Take up to the needed amount or what's left to reach 40 total
                        take_count = min(actual_shortage, available_comp, total_target - len(final_tracks))
                        
                        try:
                            compensation = random.sample(all_songs_by_type[comp_type], take_count)
                            final_tracks.extend(compensation)
                            # Remove selected tracks from the pool
                            for track in compensation:
                                all_songs_by_type[comp_type].remove(track)
                            
                            actual_shortage -= take_count
                            print(f"Compensated {take_count} tracks from {comp_type} for {type_name}")
                        except ValueError as e:
                            print(f"Error sampling from {comp_type}: {e}")
                        
                        # If shortage is resolved or we have 40 songs, break out
                        if actual_shortage <= 0 or len(final_tracks) >= total_target:
                            break
        
        # If we still don't have enough tracks, do a direct track search
        if len(final_tracks) < total_target:
            remaining = total_target - len(final_tracks)
            print(f"Still short by {remaining} tracks, doing general search")
            
            # Direct track search as fallback
            fallback_query = f"{emotion} {FALLBACK_KEYWORDS}"
            try:
                fallback_results = sp.search(q=fallback_query, type='track', limit=remaining * 2, market='IN')
                if fallback_results and 'tracks' in fallback_results and 'items' in fallback_results['tracks']:
                    fallback_tracks = _process_tracks(fallback_results['tracks']['items'], is_search=True)
                    
                    for track in fallback_tracks:
                        if len(final_tracks) >= total_target:
                            break
                        if track['id'] not in track_ids:
                            track_ids.add(track['id'])
                            final_tracks.append(track)
            except Exception as e:
                print(f"Error in fallback search: {str(e)}")
        
        print(f"Final recommendation: {len(final_tracks)} tracks out of target {total_target}")
        
        # If we have too few tracks, fall back to direct emotion search
        if len(final_tracks) < 10:
            print("Too few tracks found with playlist approach. Falling back to direct search.")
            try:
                # First try with emotion songs
                original_query = f"{emotion} songs"
                original_results = sp.search(q=original_query, type='track', limit=total_target, market='IN')
                if original_results and 'tracks' in original_results and 'items' in original_results['tracks']:
                    direct_tracks = _process_tracks(original_results['tracks']['items'], is_search=True)
                    
                    # Combine with any tracks we already found
                    for track in direct_tracks:
                        if len(final_tracks) >= total_target:
                            break
                        if track['id'] not in track_ids:
                            track_ids.add(track['id'])
                            final_tracks.append(track)
                
                # If still don't have enough, try popular songs
                if len(final_tracks) < 10:
                    popular_query = "popular songs"
                    popular_results = sp.search(q=popular_query, type='track', limit=total_target, market='IN')
                    if popular_results and 'tracks' in popular_results and 'items' in popular_results['tracks']:
                        popular_tracks = _process_tracks(popular_results['tracks']['items'], is_search=True)
                        
                        for track in popular_tracks:
                            if len(final_tracks) >= total_target:
                                break
                            if track['id'] not in track_ids:
                                track_ids.add(track['id'])
                                final_tracks.append(track)
            except Exception as e:
                print(f"Error in direct search fallback: {str(e)}")
        
        # Ensure we return something even if it's empty
        if not final_tracks:
            print("EMERGENCY FALLBACK: Using predefined playlist as last resort")
            try:
                # Use a predefined playlist as last resort
                playlist_id = PLAYLIST_IDS.get(emotion, PLAYLIST_IDS['Neutral'])
                results = sp.playlist_tracks(playlist_id, limit=total_target)
                emergency_tracks = _process_tracks(results['items'])
                final_tracks = emergency_tracks[:total_target]  # Enforce limit here too
            except Exception as e:
                print(f"Emergency fallback failed: {str(e)}")
                # As an absolute last resort, return empty list
                return []
                
        print(f"Final recommendation count: {len(final_tracks)} tracks")
        
        # Final step: Shuffle the tracks for a random language distribution
        random.shuffle(final_tracks)
        print("Shuffled tracks for random language distribution")
        
        # Ensure we never exceed the target
        return final_tracks[:total_target]
        
    except Exception as e:
        print(f"Error getting dynamic recommendations: {str(e)}")
        traceback_str = traceback.format_exc()
        print(f"Traceback: {traceback_str}")
        return []

def _process_tracks(tracks, is_search=False):
    """Helper function to process track information"""
    processed_tracks = []
    for item in tracks:
        track = item['track'] if not is_search else item
        if track and track.get('album'):
            album_image = track['album']['images'][0]['url'] if track['album']['images'] else ''
            processed_tracks.append({
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'url': track['external_urls']['spotify'],
                'id': track['id'],
                'preview_url': track['preview_url'],
                'duration_ms': track['duration_ms'],
                'album_image': album_image,
                'popularity': track.get('popularity', 0)  # Add popularity for sorting
            })
    return processed_tracks

# Emotion-to-query mapping for dynamic song recommendations
emotion_query_mapping = {
    "Angry": "intense rock",
    "Disgusted": "calm and soothing",
    "Fearful": "relaxing music",
    "Happy": "upbeat bollywood",
    "Neutral": "ambient music",
    "Sad": "melancholic hindi",
    "Surprised": "party telugu"
}

def get_songs_for_emotion(emotion: str, limit: int = 10):
    """
    Fetches a list of songs from Spotify based on the detected emotion.
    
    :param emotion: The detected emotion from the emotion recognition model.
    :param limit: The number of songs to fetch (default is 10).
    :return: A list of song details (name, artist, spotify_url).
    """
    if emotion not in emotion_query_mapping:
        print(f"No query mapping found for emotion: {emotion}")
        return []
    
    query = emotion_query_mapping[emotion]
    results = sp.search(q=query, type='track', limit=limit, market='IN')

    songs = []
    for track in results['tracks']['items']:
        song_name = track['name']
        artist_name = track['artists'][0]['name']
        spotify_url = track['external_urls']['spotify']
        songs.append({
            'name': song_name,
            'artist': artist_name,
            'url': spotify_url
        })
    
    return songs

# Example usage:
# print(get_songs_for_emotion("Happy"))

def create_emotion_model(num_classes=7):
    # Load pre-trained EfficientNetV2-L
    base_model = EfficientNetV2L(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
])
