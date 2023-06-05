from flask import Flask, request
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64
import zipfile

# Load your generator model
generator = load_model("data/models/generator_15000")

app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    num_images = data.get('num_images', 1)
    images = generate_avatars(generator, num_images)
    return {"images": images}


def generate_avatars(generator, num_images, noise_dim=100):
    images = []
    for _ in range(num_images):
        img = generate_avatar(generator, noise_dim)
        images.append(img)
    return images


def generate_avatar(generator, noise_dim):
    # Generate a random noise vector
    noise = np.random.normal(0, 1, (1, noise_dim))

    # Use the generator to create a new avatar
    img = generator.predict(noise)

    # Rescale the image from [-1,1] to [0,255]
    img = (0.5 * img[0] + 0.5) * 255
    img = img.astype(np.uint8)

    # Convert the image to base64 and then decode it to a string
    img = Image.fromarray(img)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str


if __name__ == "__main__":
    app.run(port=5000, debug=True)
