import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import cv2
from collections import deque
import os
import subprocess



# loading the saved model
loaded_model = load_model("C://Users/rubic/Desktop/streamlit/VideoClassificationApp-main/Model___Date_Time_2023_01_27__11_53_40___Loss_1.492785930633545___Accuracy_0.25.h5")


# Specify the height and width to which each video frame will be resized in our dataset.
# IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64

# # Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
# CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]

# # Specify the number of frames of a video that will be fed to the model as one sequence.
# SEQUENCE_LENGTH = 20

IMAGE_HEIGHT , IMAGE_WIDTH = 128, 128

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 20

# Specify the directory containing the UCF50 dataset. 
DATASET_DIR = "C:/Users/rubic/Desktop/streamlit/VideoClassificationApp-main/weizmann_dataset"

# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = ["bend", "gallop sideways", "jump", "jump in place"]

# creating a function for Prediction
def predict_on_live_video(video_file_path, output_file_path, window_size):
    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen = window_size)
    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)
    # Getting the width and height of the video
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 24, (original_video_width, original_video_height))
    while True:
        # Reading The Frame
        status, frame = video_reader.read()
        if not status:
            break
        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = loaded_model.predict(np.expand_dims(normalized_frame, axis = 0))[0]
        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)
        # Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window_size:
            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)
            # Calculating Average of Predicted Labels Probabilities Column Wise
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)
            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)
            # Accessing The Class Name using predicted label.
            predicted_class_name = CLASSES_LIST[predicted_label]
            # Overlaying Class Name Text Ontop of the Frame
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Writing The Frame
        video_writer.write(frame)
        # cv2.imshow('Predicted Frames', frame)
        # key_pressed = cv2.waitKey(10)
        # if key_pressed == ord('q'):
        #     break
    # cv2.destroyAllWindows()
    # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them.
    video_reader.release()
    video_writer.release()

def main():  
    # giving a title
    st.title('Video Classification Web App')
    #Upload video file
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg"])
    if uploaded_file is not None:
        #store the uploaded video locally
        with open(os.path.join("C://Users/rubic/Desktop/streamlit/VideoClassificationApp-main/VideoSample/",uploaded_file.name.split("/")[-1]),"wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File Uploaded Successfully")
                       
        if st.button('Classify The Video'):
            # Construct the output video path.
            output_video_file_path = "C://Users/rubic/Desktop/streamlit/VideoClassificationApp-main/video/"+uploaded_file.name.split("/")[-1].split(".")[0]+"_output1.mp4"
            with st.spinner('Wait for it...'):
                # Perform Action Recognition on the Test Video.
                predict_on_live_video("C://Users/rubic/Desktop/streamlit/VideoClassificationApp-main/VideoSample/"+uploaded_file.name.split("/")[-1], output_video_file_path, SEQUENCE_LENGTH)
                #OpenCV’s mp4v codec is not supported by HTML5 Video Player at the moment, one just need to use another encoding option which is x264 in this case 
                os.chdir('C://Users/rubic/Desktop/streamlit/VideoClassificationApp-main/video/')
                subprocess.call(['ffmpeg','-y', '-i', uploaded_file.name.split("/")[-1].split(".")[0]+"_output1.mp4",'-vcodec','libx264','-f','mp4','output4.mp4'],shell=True)
                st.success('Done!')
            
            #displaying a local video file
            video_file = open("C:/Users/rubic/Desktop/streamlit/VideoClassificationApp-main/video/"+"output4.mp4", 'rb') #enter the filename with filepath
            video_bytes = video_file.read() #reading the file
            st.video(video_bytes) #displaying the video


            # #displaying a local video file
            # video_file = open("C://Users/rubic/Desktop/streamlit/VideoClassificationApp-main/" + 'output4.mp4', 'rb') #enter the filename with filepath
            # video_bytes = video_file.read() #reading the file
            # st.video(video_bytes) #displaying the video
    
    else:
        st.text("Please upload a video file")
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
