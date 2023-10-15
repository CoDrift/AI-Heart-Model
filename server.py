from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model("model.keras")  # yahan par jahan model save kroge woh dal do


@app.route('/analyze_ecg', methods=['POST'])
def preprocess_image():
    if 'ecg_image' not in request.files:
        return jsonify({"error": "No 'ecg_image' file provided"})

    image_data = request.files['ecg_image']

    # Preprocess the image data
    img = Image.open(io.BytesIO(image_data.read()))  # Load the image from bytes
    desired_width = 187  # Specify your desired width
    desired_height = 1  # Specify your desired height
    img = img.resize((desired_width, desired_height))
    img = img.convert('L')  # Convert to grayscale
    img_array = np.array(img)
    img_array = img_array.reshape(-1, 187)

    # Make predictions using the loaded model
    predictions = model.predict(img_array)
    predictions = (predictions > 0.5)  # Convert probabilities to binary predictions

    result = ""

    print(predictions.tolist()[0][0])

    if predictions.tolist()[0][0] == True:
        result += "Bradycardia (Slow heart rate)"
        result += "\nThis is a rudimentary diagnosis. Please consult a professional."

    elif predictions.tolist()[0][0] == False:
        result += "Normal heart rate (approximation)"
        result += "\nThis is a rudimentary diagnosis. Please consult a professional."

    else:
        result += "Image cannot be analysed\nPlease provide a ECG chart"

    # Create a response containing the predictions
    response = {
        "predictions": predictions.tolist()[0][0],
    }

    return jsonify({"result": result})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
