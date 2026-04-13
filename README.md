# Moodify: A Music Recommendation System Using Facial Emotion Recognition

## Project Description
Moodify is an intelligent, real-time music recommendation system that leverages Facial Emotion Recognition (FER) to enhance user experience by suggesting songs aligned with your current emotional state. It captures facial expressions through a live webcam feed, classifies the emotion using a fine-tuned EfficientNet deep learning model, and dynamically fetches relevant tracks using the Spotify Web API. It features seamless in-site playback via the Spotify Web Playback SDK.

## Project Structure
The project is organized into the following directory structure to handle backend logic, frontend assets, and templates:

* **static/**: Contains all UI assets and stylesheets.
    * `favicon moodify`: The application icon.
    * `logo.png`: Project branding.
    * `player.css`: Styles specifically for the music player controls.
    * `start-vinyl.png`: Placeholder image for the music player.
    * **css/**:
        * `style.css`: Main application stylesheet.
* **templates/**: Contains the frontend HTML files.
    * `index.html`: The main dashboard and landing page for the application.

## Features
* **Real-Time Emotion Detection:** Uses OpenCV and Haar Cascades for live face detection and EfficientNet for emotion classification (Happy, Sad, Angry, Surprised).
* **Smart Music Mapping:** Maps detected emotions to specific music genres and languages (e.g., Telugu, Hindi, English).
* **Integrated Spotify Player:** Listen to recommended tracks directly within the web application using the Spotify Web Playback SDK.
* **Responsive UI:** Clean, interactive interface built with HTML/CSS/JS, including Dark/Light mode toggles and an embedded music player.

## Tech Stack
* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Flask (Python)
* **Machine Learning & Computer Vision:** TensorFlow/Keras (EfficientNetV2L), OpenCV, NumPy, Pandas
* **Music Integration:** Spotipy (Spotify API Wrapper), OAuth 2.0

## Dataset
The dataset used for this project is the **FER-2013 dataset** from Kaggle. It contains approximately 35,000 grayscale facial images (48x48 pixels) labeled across 7 emotion categories. 

* **Download Link:** [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
* **Note:** To run this project locally, download the dataset from the link above and place the images in the `data/` folder.


## Project Models
This project utilizes several trained deep learning models for emotion classification. Due to GitHub's file size limits, they are hosted externally. You can download them from the links below:

* **Primary Model:** [Download EfficientNet Model (.keras)](https://drive.google.com/file/d/1lVfK3MHZAwwUNCXc39_qJP4d_m8HV8Q9/view?usp=drive_link) - Fine-tuned EfficientNetV2L.
* **Legacy/VGG16 Model:** [Download VGG16 Model (.h5)](https://drive.google.com/file/d/16astkL5xM8EIsP3vf76de5QqWFKpjCVD/view?usp=drive_link) - Alternative model for testing.
* **Standard Model:** - The standard serialized model file found in the project directory.

**Note:** After downloading, place these files in the root directory of the project for the code to run correctly. Standard Model `model.h5` is already present in the root directory.

## Running the App
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/srisaisohan29/Moodify.git](https://github.com/srisaisohan29/Moodify.git)
   cd Moodify
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up Spotify Developer Credentials:**
   * Go to the Spotify Developer Dashboard.
   * Create an app to get your `Client ID` and `Client Secret`.
   * Create a `.env` file in the root directory and add:
     ```env
     SPOTIPY_CLIENT_ID='your_client_id_here'
     SPOTIPY_CLIENT_SECRET='your_client_secret_here'
     SPOTIPY_REDIRECT_URI='[http://127.0.0.1:5000/callback](http://127.0.0.1:5000/callback)'
     ```
4. **Run the application:**
   ```bash
   python app.py
   ```

## Credits
Developed by **Valishetty Sri Sai Sohan** and **Yempati Raghavendra Swamy**.
