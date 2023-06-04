from flask import Flask, render_template, request, send_file
from flask_bootstrap import Bootstrap
from keras.models import load_model
from PIL import Image
import numpy as np
import io

# Load your generator model
generator = load_model("path_to_your_model")

app = Flask(__name__)
Bootstrap(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle form submission here
        num_images = request.form.get('num_images')

        # Generate the images
        images = generate_avatars(num_images)

        # Save the images to temporary files and send them to the client
        # You might need to modify this depending on your exact use case
        image_files = []
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            image_files.append(('image{}.png'.format(i), buf))

        return send_file(image_files, as_attachment=True)

    return render_template('index.html')

def generate_avatars(num_images):
    images = []
    for _ in range(num_images):
        img = generate_avatar(generator)
        images.append(img)
    return images

def generate_avatar(generator, noise_dim=100):
    # Generate a random noise vector
    noise = np.random.normal(0, 1, (1, noise_dim))
    
    # Use the generator to create a new avatar
    img = generator.predict(noise)
    
    # Rescale the image from [-1,1] to [0,255]
    img = (0.5 * img[0] + 0.5) * 255
    img = img.astype(np.uint8)

    # Convert the NumPy array to a PIL image so we can save it to a file
    img = Image.fromarray(img)

    return img

if __name__ == "__main__":
    app.run(port=5000, debug=True)
