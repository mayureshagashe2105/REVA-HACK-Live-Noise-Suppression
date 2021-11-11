# Noise-Suppression
In this era of online meetings, background noise removal systems are really essential for effective workflow. This project is a model that suppresses background noise.

## Understanding The Problem Statement
Nowadays, online meetings, online schooling, and work-from-home methodologies are being used extensively due to the COVID-19 pandemic! Despite the success of this Online-X, the domestication factor and lack of a professional work environment may introduce a potential barrier that can hinder the establishment of an effective workflow. We often observe annoying background noises at the speaker's end that adds up to the noise. Hence to avoid this **"STUPID BACKGROUND NOISE"**, a robust system can be employed that will suppress the background noise giving the ideal results. This project aims to provide the solution for the above-discussed problem using a deep learning approach!

## Live Test The App
[Link To The App](https://share.streamlit.io/mayureshagashe2105/reva-hack-live-noise-suppression/main/app.py)

## Model Architecture
With the help of Convolutional Autoencoders, this model is trained to detect the unwanted noise in the audio samples. An audio sample is passed through series of 1D CONV layers and then are stacked with their respective CONV1D TRANSPOSE layers to achieve initial sample length.

<img src="utils/images/model_plot.png">

## Prototype Insights

<img src="utils/images/2021-10-28 (2).png">

---

<img src="utils/images/2021-10-28 (3).png">

---

<img src="utils/images/2021-10-28 (4).png">
