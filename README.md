LLM-Denoiser
This project, "LLM-Denoiser," is a deep learning solution for audio denoising. It leverages a model, likely a type of recurrent or convolutional neural network, to remove background noise from audio signals.

In essence, the project provides a complete pipeline for training an audio denoising model and using it to clean up noisy audio recordings.

Workflow
The workflow is structured into several key scripts. Follow these steps to set up and run the project:

Install Dependencies: This command installs all the necessary Python libraries and dependencies required to run the project.

pip install -r requirements.txt

Prepare the Dataset: This script is responsible for preparing the dataset. It likely involves loading clean and noisy audio pairs, processing them into a suitable format (like spectrograms), and creating data loaders for training and testing.

python data.py

Train the Model: This is the core training script. It defines the neural network architecture, the loss function, and the optimizer. It then iterates through the prepared dataset to train the model, saving the trained model weights upon completion.

python train.py

Perform Inference: After training, this script is used for inference. It loads the saved model and applies it to new, noisy audio files to produce cleaned, denoised versions.

python infer.py

Visualize Results: This utility script is used for visualization. It generates plots, such as waveforms or spectrograms, to visually compare the original noisy audio, the clean audio, and the model's denoised output, helping to evaluate the model's performance.

python plot_audio_charts.py
