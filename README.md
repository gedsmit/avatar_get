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

`
git clone https://github.com/gedsmit/GAN_Avatar_Generator.git
cd GAN_Avatar_Generator
`

2. Run the application:

`
run_app.cmd
`

3. To train the model refer to Colab Notebook:


[Colab Notebook](https://colab.research.google.com/drive/1ZtTck-4_tR85hkRPtMhkrgeVXL4qigA4?usp=sharing)


This command will install the required packages and start the application.

## API Documentation

To generate avatars, send a POST request to the `/generate` endpoint, specifying the number of avatars you want to generate.

Example:

`
curl -X POST -d "num_images=5" http://localhost:5000/generate
`

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Reflections and Future Work

This project served as an excellent opportunity to delve deeper into GANs and learn about end-to-end deployment of deep learning models in a real-life scenario. 

The project borrows its primary model structure from a Kaggle project and uses pretrained weights. The training process proved to be challenging due to the complex nature of GANs, reminding us of the intricate balance needed to effectively train such models. We modified the discriminator model by incorporating a pretrained VGG16 model, which improved the model's performance.

The use of Google Colab presented a hurdle due to its limitations, which reinforced the importance of robust hardware for deep learning tasks. Despite these challenges, we were able to train the model for 200 epochs.

In the future, we plan to continue enhancing the model's performance and training it for more epochs. We also plan to experiment with different GAN architectures and training techniques to improve the quality of the generated avatars. We look forward to updating our model and delivering even better results in avatar generation.
