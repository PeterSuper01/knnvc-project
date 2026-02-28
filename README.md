## Introduction
This project uses KNN-VC to convert one character's voice to another. You need one source audio from the character you want to convert and a matching audio as a reference. There are two AI models in this project, one **encoder** to extract features from the audio, and a **vocoder** to translate the features into an audio.

First, we convert the source audio and matching audio into two sets of 1024-dimensional feature vectors. Then, for each source vector, we find its k nearest neighbors among the matching vectors and average them. Finally, we convert these averaged vectors back into audio.

### WavLM encoder

The original paper used pretrained WavLM-Large as the encoder / feature extractor, and take the 6th layer in WavLM-Large as the extracted feature vectors.

This model takes audio sampled at 16kHz as input and splits it into 20-ms frames. Each frame is then transformed by the model into a 1024-dimensional feature vector.

### HiFi-GAN vocoder

The original paper trained a HiFi-GAN V1 generator to accept the 1024-dimensional input vectors
from WavLM and to vocode 16kHz audio.

Here I load the model with trained weights from [bshall/knn-vc](https://github.com/bshall/knn-vc) repository.

## Usage

Place your `.wav` files in the `data/` directory, update the `.env` file with the appropriate settings, and then execute `main.py` to run the application.

## References

* **kNN-VC Implementation**: [bshall/knn-vc](https://github.com/bshall/knn-vc)
* **Pre-trained Weights**: Loaded from `torch.hub.load('bshall/knn-vc', 'hifigan_wavlm')`
* **Original Paper**: Baas et al., "[Voice Conversion With Just Nearest Neighbors](https://arxiv.org/abs/2305.18975)," *Interspeech 2023*.