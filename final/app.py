import os
import random
import cv2
import matplotlib.pyplot as plt
from flask import Flask, render_template
from markupsafe import Markup

 # Import Markup from flask.wrappers

# Initialize Flask app
app = Flask(__name__)

# Initialize lists to store image names and predicted text
image_names = []
predicted_texts = []

# Number of images to generate and predict
num_images = 10
match_count = 0

# Generate and predict random images
for i in range(num_images):
    samples_folder = "C:/Users/SHANKAR P/Downloads/Captcha-recognition-using-CNN-master/Captcha-recognition-using-CNN-master/captcha_dataset (2)/samples"
    image_files = [f for f in os.listdir(samples_folder) if f.endswith('.png')]
    random_image = random.choice(image_files)
    img_name = os.path.splitext(random_image)[0]
    img = cv2.imread(os.path.join(samples_folder, random_image), cv2.IMREAD_GRAYSCALE)
    if img is not None:
         plt.imshow(img, cmap=plt.get_cmap('gray'))
         plt.axis('off')  # Hide axis
         plt.savefig('static/img.png', bbox_inches='tight', pad_inches=0)  # Save image without extra whitespace
         plt.close()
         predicted_text = predict(os.path.join(samples_folder, random_image))
         image_names.append(img_name)
         predicted_texts.append(predicted_text)
         match_count += 1 if img_name == predicted_text else 0

@app.route('/')
def index():
    return render_template('index.html', num_images=num_images, accuracy=(match_count / num_images) * 100)

if __name__ == '__main__':
    app.run(debug=True)
