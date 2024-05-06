
import cv2
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import altair as alt
import cv2 as cv
import tempfile
import joblib
import speech_recognition as sr
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

pipe_lr = joblib.load(open("Text_Modal/text_emotion.pkl", "rb"))


emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", 
                    "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: 
"Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("emotion_model.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")



def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img





def speech_to_text():


    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Create the start recording button
    start_recording_button = st.button("Start Recording")

    # Recording status indicator
    recording_status = st.empty()

    if start_recording_button:
        recording_status.write("Recording...")

        # Start recording audio from the user
        with st.spinner("Listening..."):
            with sr.Microphone() as source:
                st.write("Please speak something...")
                audio_data = recognizer.listen(source)

        recording_status.write("Recording completed!")

        # Convert audio to text
        st.subheader("Converted Text:")
        try:
            text_result = recognizer.recognize_google(audio_data, language="en-US")
            st.write(text_result)

            col1, col2 = st.columns(2)

            prediction = predict_emotions(text_result)
            probability = get_prediction_proba(text_result)


            with col1:
                st.success("Original Text")
                st.write(text_result)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))


            with col2:
                st.success("Prediction Probability")
            #st.write(probability)
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            #st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)




        except sr.UnknownValueError:
            st.error("Could not understand audio. Please try again.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")


def main():
    st.title("Multi-Modal Emotion Detection")


    selectted=option_menu(
        menu_title="Select One",
        options=["Text Emotion Detection","Video Emotion Detection","Audio Emotion Detection"],
        icons=["keyboard","camera","mic"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",)

    if selectted=="Text Emotion Detection":
        with st.form(key='my_form'):
            st.subheader("Detect Emotions In Text")
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))


            with col2:
                st.success("Prediction Probability")
            #st.write(probability)
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            #st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

    elif selectted=="Video Emotion Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)


    elif selectted=="Audio Emotion Detection":
        st.subheader("Detect Emotions In Audio")
        speech_to_text()

    else:
        pass







if __name__ == '__main__':
    main()
