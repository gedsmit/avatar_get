# GAN Avatar Generator API

This repository contains an API for generating unique avatar images using a trained GAN (Generative Adversarial Network) model.

## Description

This project provides a simple way to generate unique avatars using a pre-trained deep learning model. The model was trained on a dataset of anime faces, so the generated avatars have an anime-like appearance. The avatar generation is exposed as an API and can be easily integrated into any project.

## Features

- REST API for generating avatar images.
- Flask web server.
- Interactive web app using Streamlit.

## Requirements

- Python 3.7+
- Flask
- Streamlit
- Keras
- TensorFlow
- PIL (Python Imaging Library)
- Numpy

## Installation

1. Clone the repository:

\`\`\`
git clone https://github.com/gedsmit/GAN_Avatar_Generator.git
cd GAN_Avatar_Generator
\`\`\`

2. Install the requirements:

\`\`\`
pip install -r requirements.txt
\`\`\`

3. Download the pre-trained model and place it in the `models/` directory.

## Usage

Start the Flask server:

\`\`\`shell
python app.py
\`\`\`

Start the Streamlit app:

\`\`\`shell
streamlit run app.py
\`\`\`

## API Documentation

To generate avatars, send a POST request to the `/generate` route with the number of avatars you want to generate.

Example:

\`\`\`shell
curl -X POST -d "num_images=5" http://localhost:5000/generate
\`\`\`

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
