
# EMG Signal Generation for data inpainting

## Overview
This project is focused on processing Electromyography (EMG) signals for data inpainting using Wasserstein Generative Adversarial Networks (wGANs). The process involves breaking down EMG signals into small sliding windows, converting them to Short-Time Fourier Transform (STFT) representations in the `data_processing` folder, and feeding these into a wGAN, located in the `models` folder, to generate new signals.

## Repository Structure

```
emg-cGAN-Project/
│
├── models/
│   ├── generator.py
│   ├── discriminator.py
│   └── [other model files]
│
├── data_processing/
│   ├── stft_conversion.py
│   └── [additional data processing scripts]
│
├── dataset.py
├── train.py
├── README.md
└── requirements.txt
```

## Key Components

### `dataset.py`
- Manages the loading and preprocessing of EMG signal data.
- Likely includes methods for converting EMG data into STFT representations or other necessary preprocessing steps.

### `train.py`
- Contains the training loop for the wGAN.
- Handles the setting up and training of discriminator and generator models from the `models` folder.
- Utilizes `dataset.py` for accessing preprocessed EMG data.

### `data_processing/stft_conversion.py`
- Dedicated script for converting EMG signals into STFT representations.
- Can be used independently for preprocessing steps.

### `models/`
- Contains model definitions for the wGAN, including both the generator and discriminator.

## Installation and Dependencies
- Python (version specified in `requirements.txt`)
- Dependencies: `torch`, `torchvision`, `numpy`, etc.
  ```
  pip install -r requirements.txt
  ```

## Usage
- Use scripts in `data_processing` to convert EMG signals to STFT.
- `dataset.py` manages the dataset for training.
- Execute `train.py` to train the wGAN on the preprocessed data.

## Contributing
Contributions to improve and extend the functionality of this project are welcome. Please adhere to coding standards and document new features or improvements adequately.

## License
University of Edinburgh 2022
