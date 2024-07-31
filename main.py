import cv2
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import moviepy.editor as mp
import os
import tempfile

app = FastAPI()

# Function to load labels from labels.txt
def load_labels(label_file_path):
    with open(label_file_path, 'r') as file:
        labels = [line.strip() for line in file.readlines()]
    return labels

# Load labels from the specified file
labels = load_labels("E:/New folder/labels.txt")  # Ensure this path matches your directory structure

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application"}

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    try:
        # Read the video file into memory
        video_bytes = await video.read()
        
        # Create a temporary file for the video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_bytes)
            temp_file_path = temp_file.name

        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path="E:/New folder/model_unquant.tflite")
        interpreter.allocate_tensors()

        # Get model input details
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        input_shape = input_details['shape']
        input_height, input_width = input_shape[1], input_shape[2]

        # Initialize video capture from temporary file using moviepy
        with mp.VideoFileClip(temp_file_path) as video_clip:
            results = []

            for frame in video_clip.iter_frames(fps=video_clip.fps, dtype='uint8'):
                # Preprocess frame for model
                frame_resized = cv2.resize(frame, (input_width, input_height))
                frame_normalized = frame_resized.astype(np.float32) / 255.0
                frame_input = np.expand_dims(frame_normalized, axis=0)

                # Run inference
                interpreter.set_tensor(input_details['index'], frame_input)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details['index'])

                # Map the output to labels
                predicted_class = np.argmax(output[0])
                result_label = labels[predicted_class]
                results.append(result_label)

        # Clean up temporary file
        os.remove(temp_file_path)

        # Calculate the most frequent label
        unique, counts = np.unique(results, return_counts=True)
        summary = dict(zip(unique, counts))
        most_frequent_label = max(summary, key=summary.get)

        return {"most_frequent_label": most_frequent_label}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
