# GANs-for-Medical-Data
## Description
This repo was used for research from August 2021 to May 2022. I focused on applying generative adversarial networks (GANs) to time series medical data, specifically the [Wearable Stress and Affect Detection (WESAD) dataset](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29), and some work done with the [MIT-BIH dataset](https://physionet.org/content/mitdb/1.0.0/). This repo primarily contains data processing code for the WESAD dataset.

## Problem Description
Machine learning models are data hungry, meaning they require a vast amount of diverse training data for the model to generalize well and perform accurately. In the medical domain, this data can also have privacy issues - if the data is sensitive or the patient would not like it shared with others, then it shouldn't be used in machine learning models. Additionally, sensitive medical data could be exposed through membership inference attacks. 

## Approach
Synthetic data could resolve this issue, allowing machine learning practitioners to generate data that follows a similar distribution to actual data but is different enough to expose sensitive information from patients. This is why I tried using GANs to generate synthetic data. GANs have shown strong results with generating realistic images, and are being explored further for other modes of data, such as time series signals. I chose to focus on the WESAD dataset since it contained multi-modal time series data. I started with state-of-the-art models and began applying them to the WESAD dataset's electrocardiogram (ECG) data. I found varying success with different models, but found a BiLSTM GAN model that could generate synthetic data in a similar distribution to that of the training data. I experimented with different preprocessing approaches to find an optimal way to select training data and denoise the signal, described in the next section.

## Data Preprocessing
The WESAD dataset contains raw time series data for 15 subjects. For generation of synthetic data, it was easiest to split up the time series data into small chunks, which would each be a data point / sample in our dataset. To do this, a naive approach could be picking a fixed time length and splitting up the data into chunks that long. However, this medical data is highly dependent on time. An electrocardiogram (ECG) signal contains different sections and peaks that repeat. Therefore, it makes sense to split the data into the individual signals that repeat. To do this, we can extract the location of the R-peaks for the ECG signals, and use that to split the data.

Additionally, the data suffers from noise. I experimented with de-noising the data using a moving average and other types of filters. I found that smoothing with a moving average filter improved the performance of the GAN (as measured by the maximum mean discrepancy, see below) compared with when using the unfiltered signals. Below are different examples of preprocessing I experimented with: 

## Results
### Evaluation Metrics
To evaluate the performance of synthetic data, there are a variety of methods:
- [Maximum Mean Discrepancy](http://www.gatsby.ucl.ac.uk/~gretton/papers/cardiff.pdf) can compare the difference between two distributions of data. We can compare the distribution of real data to that of the generated synthetic data to see if the synthetic data accurately captures the distribution of the real data.
- Train Synthetic, Test Real (TSTR). For this method, we first train a classifier to perform a certain task using the real data. We then evaluate its performance on a testing set of real data. We then train a classifier on the same task but using synthetic data generated by the GAN. We then evaluate this classifier's performance on the same testing set of real data. If the distribution of the synthetic data is similar to that of the real data the GAN was trained on, then the two classifiers should perform about the same.
- [Other metrics described here](https://ieeexplore.ieee.org/document/9534373). Note that this paper found TSTR is more efficient that TRTS.

To test if the GAN could accurately capture the nuances in the ECG data between each subject, I performed a TSTR experiment where the classifier's task was to guess which subject the data came from. If the GAN can capture these nuances, then the classifier should perform about as well when trained on the synthetic data versus when trained on the real data. When performing the TSTR experiment for classifying which subject the ECG data came from, I found that the GAN could achieve 99% accuracy when trained and tested on real data, and 95% accuracy when trained on synthetic data and tested on real data.

## Literature Review
I completed a literature review in August 2021 on the then current state of applications for GANs in time series data, specifically medical time series data. After reviewing countless papers, I determined the research gaps to be:
- Incorporating self attention in GANs for medical data
- Incorporating simulators into generator
- Generation of non-ECG medical data like those present in the WESAD dataset
- Differential privacy training without significant reduction in performance
- No work done using GANs to generate data from WESAD dataset

I chose to focus on using GANs to generate data in a similar format to that in the WESAD dataset. This would be a novel contribution since it would involve both 1) the generation of time series non-ECG medical data (such as blood pressure, electrodermal activity, electromyogram, etc.), and 2) using a GAN to generate multi-model medical time series data.
