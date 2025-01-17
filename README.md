# Encoding Methods Comparison for Stress Detection

## Abstract

Stress is a prevalent and growing phenomenon in the modern world that could lead to significant physical issues, both
physical and mental health. Analyzing physiological signals collected from wearable sensors using artificial
intelligence methods has emerged as a promising approach to predicting and managing stress. However, conventional models
for time series analysis are RNN architectures and encounter challenges like high computational costs and issues with
vanishing or exploding gradients. Inspired by the success of deep learning methods in computer vision, several studies
have proposed transforming time series into images by applying encoding time series algorithms.

This work compares three time-series encoding methods: Gramian Angular Field (GAF), both summation and difference,
Markovian Transition Field (MTF) and Recurrent Plot (RP) in the stress detection scenario. We employ two architectures,
VGG-16 and ResNet, based on Convolutional Neural Network (CNN), to evaluate the performance of these methods on a public
dataset, WESAD. Our results demonstrate that the GAF encoding method proves to be the most effective for classifying
physiological signals related to stress. For ResNet-18, the GAF-s method excels, while for VGG-16, the GAF-d method
shows better performance, achieving an accuracy of 95.8% and a weighted F1-score of 93.17%.

This research was published in the proceedings of the 3rd AIxIA Workshop on Artificial Intelligence For Healthcare and
5th Data4SmartHealth, November 25-28, 2024, Bolzano, Italy.
The research paper can be accessed [here](https://ceur-ws.org/Vol-3880/paper27.pdf).

## Dataset

The WESAD (Wearable Stress and Affect Detection) dataset is a multimodal dataset featuring lab-sourced data from 15
participants wearing sensors that record various physiological signs including:

- Blood volume pulse (BVP)
- ECG
- Electrodermal activity (EDA)
- Electromyogram (EMG)
- Respiration (RESP)
- Body temperature (TEMP)
- Three-axis acceleration (ACC)

The data is collected through both chest-mounted (RespiBan) and wrist-worn (Empatica E4) devices, sampled at 700 Hz and
35 Hz respectively.

## Installation

To install the required packages, you can use the `requirements.txt` file. Run the following command:

```bash
pip install -r requirements.txt
```

## Running the Code

You can run the main script with various options. Here is an example of how to use the script:

```bash
python main.py --freq 700 --dataset data/WESAD --sec 60 --window_stride 1 --patience 10 --epochs 100 --methods gaf_difference gaf_summation mtf rp_euclidean rp_dtw --model custom --labels 1 2
```

### Command Line Arguments

- `--freq`: Frequency of the preprocessing (default: 700)
- `--dataset`: Path to the dataset (default: 'data/WESAD')
- `--sec`: Window in seconds (default: 60)
- `--window_stride`: Window stride in seconds (default: 1)
- `--patience`: Early stopping patience (default: 10)
- `--epochs`: Number of epochs (default: 100)
- `--methods`: Methods to run (default: ['gaf_difference', 'gaf_summation', 'mtf', 'rp_euclidean', 'rp_dtw'])
- `--model`: Model choice for training (choices: ['custom', 'vgg', 'resnet'], default: 'custom')
- `--labels`: List of labels for classification (default: [1, 2])

## Project Structure

- `preprocessing.py`: Contains functions for preprocessing the data.
- `encoding/`: Contains the different encoding techniques (GAF, MTF, RP, RP-DTW).
- `model/`: Contains the CNN models used in this project.
- `utils/`: Contains utility functions for email notifications, plotting, etc.
- `main.py`: The main script for running the experiments.

## Results

The results of the experiments are saved in the `Results/` directory. Plots and model checkpoints are saved for each
experiment run.

## Contact

For any questions or issues, please contact Marco Serenelli on Github or Linkedin.
