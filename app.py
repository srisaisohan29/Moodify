from flask import Flask, render_template, redirect, url_for, session, request, jsonify
from camera import detect_emotion
from Spotipy import create_spotify_oauth, get_token, get_playlist_tracks, get_dynamic_recommendations
import time
import os
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

@app.route('/')
def index():
    if not session.get('token_info', False):
        return redirect(url_for('login'))
    return render_template('index.html', songs=[], current_song=None)

@app.route('/login')
def login():
    sp_oauth = create_spotify_oauth()
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    sp_oauth = create_spotify_oauth()
    session.clear()
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    session["token_info"] = token_info
    return redirect(url_for('index'))

@app.route('/detect')
def detect():
    if not session.get('token_info', False):
        return redirect(url_for('login'))
    emotion = detect_emotion()
    session['emotion'] = emotion
    return redirect(url_for('recommend'))

@app.route('/recommend')
def recommend():
    if not session.get('token_info', False):
        return redirect(url_for('login'))
    
    emotion = session.get('emotion')
    if emotion:
        songs = get_playlist_tracks(emotion)
    else:
        songs = []
    
    return render_template('index.html', songs=songs, current_song=songs[0] if songs else None, emotion=emotion)

@app.route('/get_mood_playlist')
def get_mood_playlist():
    emotion = request.args.get('emotion')
    if not emotion:
        return jsonify({'songs': []})
    songs = get_playlist_tracks(emotion)
    return jsonify({'songs': songs})

@app.route('/get_mix_playlist')
def get_mix_playlist():
    emotion = request.args.get('emotion')
    if not emotion:
        return jsonify({'songs': []})
    songs = get_dynamic_recommendations(emotion)
    return jsonify({'songs': songs})

@app.route('/token')
def get_spotify_token():
    try:
        token = get_token()
        return {'token': token}
    except:
        return redirect(url_for('login'))

def check_token():
    token_info = session.get("token_info", {})
    now = int(time.time())
    is_expired = token_info.get('expires_at', now) < now

    if is_expired:
        sp_oauth = create_spotify_oauth()
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
        session['token_info'] = token_info

if __name__ == '__main__':
    app.run(debug=True)

