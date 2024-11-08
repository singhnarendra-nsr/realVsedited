from flask import Flask, render_template, request
import cv2
import pickle

# Load the model
with open('cnn_model3.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    # Load the image using OpenCV
    test_img = cv2.imread(image_path)

    # Resize the image
    test_img = cv2.resize(test_img, (256, 256))

    # Prepare the image for the model
    test_input = test_img.reshape((1, 256, 256, 3))

    # Make a prediction using the model
    a = model.predict(test_input)
    a = a[0][0]

    # Calculate the percentage for real and edited
    real_percentage = round(a * 100, 2)  # Percentage of being a real image
    edited_percentage = round((1 - a) * 100, 2)  # Percentage of being an edited image

    # Determine the classification
    if a > 0.5:
        classification = f"Real Image ({real_percentage}%)"
    else:
        classification = f"Edited Image ({edited_percentage}%)"

    # Pass the information to the template
    return render_template('index.html', prediction=classification)


import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
