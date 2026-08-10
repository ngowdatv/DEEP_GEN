# Deep Generative Framework for Synthetic Data Generation

## 1. Project Overview

The Deep Generative Framework is a modular synthetic data generation system designed to generate realistic synthetic data from different types of datasets.

The framework is designed to support:

- Image datasets
- CSV/tabular datasets
- Text datasets

Each dataset type can be processed using an appropriate generative model.

## 2. Project Objective

The main objective of this project is to develop a unified framework that accepts a user dataset, identifies the dataset type, applies the appropriate generative model, and produces synthetic data.

The overall workflow is:

User Dataset
        |
        v
Dataset Type Detection
        |
        +----------------+----------------+
        |                |                |
      Images             CSV             Text
        |                |                |
       GAN              CTGAN        Text Generation
        |
        v
Synthetic Data
        |
        v
Evaluation
        |
        v
Frontend Display

## 3. Team Responsibilities

| Team Member | Responsibility |
|-------------|----------------|
| Member 1 | GAN-based image generation |
| Member 2 | VAE / additional image generation |
| Member 3 | CTGAN for CSV datasets |
| Member 4 | Text generation module |
| Member 5 | Streamlit frontend and integration |

## 4. GAN Module

The GAN module is responsible for synthetic image generation.

The GAN consists of two main neural networks:

### Generator

The Generator creates synthetic images from random noise.

Input:
Random latent vector

Output:
Synthetic image

### Discriminator

The Discriminator determines whether an image is real or generated.

Input:
Image

Output:
Probability of the image being real

The two networks are trained together so that the Generator gradually improves its ability to create realistic images.

## 5. GAN Workflow

The implemented GAN pipeline follows these steps:

1. Load image dataset
2. Preprocess images
3. Build Generator
4. Build Discriminator
5. Combine both models into GAN
6. Train Generator and Discriminator
7. Save trained models
8. Generate synthetic images
9. Evaluate generated images

The workflow is:

Image Dataset
     |
     v
Image Preprocessing
     |
     v
Generator + Discriminator
     |
     v
GAN Training
     |
     v
Trained Generator
     |
     v
Synthetic Images
     |
     v
Evaluation

## 6. Project Structure

```text
Deep_Generative_Framework/
|
├── dataset/
|   ├── images/
|   ├── csv/
|   └── text/
|
├── preprocessing/
|   └── preprocess.py
|
├── models/
|   ├── generator.py
|   ├── discriminator.py
|   └── gan.py
|
├── training/
|   └── train.py
|
├── evaluation/
|   ├── metrics.py
|   ├── evaluate.py
|   └── loss_graph.py
|
├── utils/
|   ├── helper.py
|   ├── generate.py
|   └── dataset_detector.py
|
├── generated_images/
|
├── saved_models/
|
├── app.py
├── requirements.txt
└── README.md
## 7. Dataset Detection

The framework includes a dataset detection module.

The detector examines the uploaded dataset and identifies the data format based on the files present.

Supported image extensions include:

- JPG
- JPEG
- PNG
- BMP
- GIF
- WEBP

CSV datasets are identified using the CSV extension.

Text datasets can be identified using supported text-based extensions.

The detected dataset type can then be directed to the corresponding generative model.

For example:

Medical X-ray images -> Image dataset -> GAN

Sports images -> Image dataset -> GAN

Medical CSV -> CSV dataset -> CTGAN

Sports CSV -> CSV dataset -> CTGAN

Medical text -> Text dataset -> Text generation

Sports text -> Text dataset -> Text generation

The model selection is based on the dataset format rather than a fixed domain.

## 8. GAN Training

The GAN training module loads images from:

dataset/images/

The images are resized to the required dimensions and normalized before training.

The Generator and Discriminator are optimized using adversarial training.

The current implementation has been tested using a small image dataset to verify the complete pipeline.

## 9. Synthetic Image Generation

After training, the saved Generator can be used to generate synthetic images.

Generated images are stored in:

generated_images/

Example:

generated_images/
├── synthetic_1.png
├── synthetic_2.png
├── synthetic_3.png
├── ...
└── synthetic_20.png

## 10. Current Test Result

The GAN pipeline was successfully tested with an image dataset containing 4 sample images.

The pipeline successfully completed:

- Dataset loading
- GAN training
- Model generation
- Synthetic image generation

The test generated:

20 synthetic images

This confirms that the end-to-end GAN pipeline is functioning.

The small dataset is currently used for pipeline testing. Larger datasets are required for meaningful image quality evaluation and better synthetic results.

## 11. Evaluation

The project contains evaluation modules for analyzing generated images.

The evaluation stage is intended to measure the quality and similarity of generated data.

Future evaluation can include metrics such as:

- FID
- SSIM
- Image similarity
- Training loss
- Generator loss
- Discriminator loss

## 12. Frontend Integration

The Streamlit frontend will provide the user interface for uploading datasets and displaying generated results.

The intended workflow is:

Upload Dataset
        |
        v
Detect Dataset Type
        |
        v
Select Generative Model
        |
        v
Generate Synthetic Data
        |
        v
Display Results

For image datasets, the frontend will connect to the GAN module.

The GAN module will process the uploaded images and return the generated synthetic images for display in the frontend.

## 13. Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Streamlit
- Git
- GitHub

## 14. Current Status

The GAN image generation module has been implemented and tested.

Completed components:

- GAN Generator
- GAN Discriminator
- GAN architecture
- Image preprocessing
- GAN training pipeline
- Synthetic image generation
- Model saving
- Dataset type detection
- Evaluation module

The next stage is integration with the Streamlit frontend and testing with larger real-world image datasets.

## 15. Future Improvements

Future improvements include:

- Support for larger datasets
- Improved GAN architectures
- Improved synthetic image quality
- FID and SSIM evaluation
- Automatic dataset processing
- Frontend integration
- Support for multiple image formats
- Training visualization
- Improved model performance
- Complete end-to-end automated synthetic data generation