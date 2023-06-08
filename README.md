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
`

`
cd GAN_Avatar_Generator
`

2. Run the application:

`
run_app.cmd
`

3. To train the model refer to the [Colab Notebook](https://colab.research.google.com/drive/1ZtTck-4_tR85hkRPtMhkrgeVXL4qigA4?usp=sharing)


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

This project provided us with an invaluable opportunity to delve deeper into GANs and gain practical experience in deploying deep learning models in real-life scenarios.

Initially, we encountered challenges due to the limitations of Google Colab. However, after investing time and effort into addressing these issues, we were able to train our model for an extended period of 1000 epochs. Unfortunately, despite the increased training duration, the results did not meet our expectations in terms of avatar quality.

These results highlight the complexity and sensitivity of GANs, where achieving optimal performance requires careful tuning of various parameters and architectural choices. It is evident that further research and experimentation are necessary to generate avatars of higher quality.

In future work, we plan to explore alternative architectures and training techniques to enhance the performance of our model. This could involve investigating advanced GAN variants, such as Progressive GANs or StyleGAN, which have shown promising results in generating realistic and diverse images.

Additionally, we aim to address the limitations of our current approach by exploring strategies such as data augmentation, regularization techniques, or adjusting the loss functions. These techniques have the potential to further improve the quality of the generated avatars and increase their resemblance to real anime faces.

By continuing our research and refining the model, we hope to achieve better results and contribute to the advancement of avatar generation techniques. We acknowledge that there is still much to learn in this field, and we are excited about the future possibilities and the potential impact of our work.