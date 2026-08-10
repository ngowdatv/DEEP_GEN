# Deep Generative Framework for Synthetic Data Generation

## 1. Project Overview

The Deep Generative Framework for Synthetic Data Generation is a modular synthetic data generation system designed to generate synthetic data from different types of datasets.

The framework supports:

- Image datasets
- CSV/tabular datasets
- Text datasets

Each dataset type can be processed using an appropriate generative model.

---

## 2. Project Objective

The main objective of this project is to develop a unified framework that accepts a user dataset, identifies the dataset type, applies the appropriate generative model, and produces synthetic data.

### Overall Workflow

User Dataset
        |
        v
Dataset Type Detection
        |
        +----------------+----------------+
        |                |                |
        v                v                v
     Images             CSV             Text
        |                |                |
        v                v                v
       GAN              CTGAN       Text Generation
        |
        v
Synthetic Data
        |
        v
Evaluation
        |
        v
Frontend Display

---

## 3. Team Responsibilities

| Team Member | Responsibility |
|-------------|----------------|
| Member 1 | GAN-based image generation |
| Member 2 | VAE / additional image generation |
| Member 3 | CTGAN for CSV datasets |
| Member 4 | Text generation module |
| Member 5 | Streamlit frontend and integration |

---

## 4. GAN Module

The GAN module is responsible for synthetic image generation.

A GAN consists of two main neural networks:

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
Probability indicating whether the image is real or generated.

The Generator and Discriminator are trained together through adversarial training so that the Generator gradually improves its ability to create realistic images.

---

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

---

## 6. Project Structure

Deep_Generative_Framework/
|
├── dataset/
│   ├── images/
│   ├── csv/
│   └── text/
|
├── preprocessing/
│   └── preprocess.py
|
├── models/
│   ├── generator.py
│   ├── discriminator.py
│   └── gan.py
|
├── training/
│   └── train.py
|
├── evaluation/
│   ├── metrics.py
│   ├── evaluate.py
│   └── loss_graph.py
|
├── utils/
│   ├── helper.py
│   ├── generate.py
│   └── dataset_detector.py
|
├── generated_images/
|
├── saved_models/
|
├── app.py
├── requirements.txt
└── README.md

---

## 7. Dataset Detection

The framework includes a dataset detection module.

The detector examines the uploaded dataset and identifies the data format based on the files present.

### Supported Image Formats

- JPG
- JPEG
- PNG
- BMP
- GIF
- WEBP

CSV datasets are identified using the .csv extension.

Text datasets can be identified using supported text-based extensions.

The detected dataset type can then be directed to the corresponding generative model.

Examples:

Medical X-ray Images -> Image Dataset -> GAN

Sports Images -> Image Dataset -> GAN

Medical CSV -> CSV Dataset -> CTGAN

Sports CSV -> CSV Dataset -> CTGAN

Medical Text -> Text Dataset -> Text Generation

Sports Text -> Text Dataset -> Text Generation

The model selection is based on the dataset format rather than a fixed domain.

---

## 8. GAN Training

The GAN training module loads images from:

dataset/images/

The images are resized to the required dimensions and normalized before training.

The Generator and Discriminator are optimized using adversarial training.

The current implementation has been tested using a small image dataset to verify the complete pipeline.

### Training Configuration

- Latent Dimension: 100
- Batch Size: 32
- Epochs: 5
- Image Size: 128 x 128
- Image Channels: 3

---

## 9. Generator Model

The Generator is responsible for transforming a random latent vector into a synthetic image.

The current Generator uses:

- Dense layer
- Batch Normalization
- LeakyReLU activation
- Reshape layer
- Conv2DTranspose layers
- Final RGB image output

The Generator produces images with an output shape of:

128 x 128 x 3

This represents an RGB image.

---

## 10. Discriminator Model

The Discriminator is responsible for classifying images as real or generated.

The current Discriminator uses:

- Convolutional layers
- LeakyReLU activation
- Dropout
- Flatten layer
- Dense output layer

The final output represents the discriminator prediction for the input image.

---

## 11. GAN Architecture

The Generator and Discriminator are combined into the complete GAN architecture.

Random Noise
        |
        v
Generator
        |
        v
Synthetic Image
        |
        v
Discriminator
        |
        v
Real / Generated Prediction

During training, the Generator attempts to create realistic images while the Discriminator attempts to distinguish real images from generated images.

---

## 12. Synthetic Image Generation

After training, the saved Generator can be used to generate synthetic images.

Generated images are stored in:

generated_images/

Example:

generated_images/
├── synthetic_1.png
├── synthetic_2.png
├── synthetic_3.png
├── synthetic_4.png
├── synthetic_5.png
├── ...
└── synthetic_20.png

---

## 13. Current Test Result

The GAN pipeline was successfully tested with an image dataset containing 4 sample images.

The pipeline successfully completed:

- Dataset loading
- Image preprocessing
- GAN model initialization
- GAN training
- Synthetic image generation

Test Result:

Generated Images: 20

The current 4-image dataset was used for pipeline testing. Larger datasets are required for meaningful image-quality evaluation and better synthetic results.

---

## 14. Evaluation

The project contains evaluation modules for analyzing generated images.

The evaluation stage is intended to measure the quality and similarity of generated data.

Future evaluation can include:

- FID
- SSIM
- Image similarity
- Training loss
- Generator loss
- Discriminator loss

These metrics can be used to analyze generated image quality and monitor GAN training performance.

---

## 15. Dataset Independence

The GAN image-generation module is designed to be domain-independent.

The GAN can be used for different image domains as long as the input dataset contains image data.

Examples:

Medical Images -> GAN -> Synthetic Medical Images

Sports Images -> GAN -> Synthetic Sports Images

Manufacturing Images -> GAN -> Synthetic Manufacturing Images

Therefore, the GAN module is not restricted to only medical datasets.

---

## 16. Frontend Integration

The Streamlit frontend will provide the user interface for uploading datasets and displaying generated results.

### Intended Workflow

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

---

## 17. Technologies Used

The project uses:

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

---

## 18. Current Status

The GAN image generation module has been implemented and tested.

### Completed Components

- GAN Generator
- GAN Discriminator
- GAN architecture
- Image preprocessing
- GAN training pipeline
- Synthetic image generation
- Model saving
- Dataset type detection
- Evaluation module

### Current Result

The GAN pipeline successfully:

Loads Image Dataset
        |
        v
Preprocesses Images
        |
        v
Builds Generator
        |
        v
Builds Discriminator
        |
        v
Combines GAN
        |
        v
Trains GAN
        |
        v
Generates Synthetic Images
        |
        v
Produces 20 Test Images

The core GAN image-generation pipeline is currently functional.

---

## 19. Limitations

The current implementation has been tested using a small dataset containing only 4 images.

This dataset size is sufficient for testing the software pipeline but is not sufficient for evaluating the real-world quality of GAN-generated images.

For better results, the system should be tested with a larger and more diverse image dataset.

The current implementation uses CPU-based TensorFlow execution on the development environment.

---

## 20. Future Improvements

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
- Hyperparameter optimization
- Better model checkpointing
- Configurable number of generated images
- Complete end-to-end automated synthetic data generation

---

## 21. Conclusion

The GAN component provides the image generation module of the Deep Generative Framework.

The current implementation demonstrates the complete image-generation pipeline, from loading and preprocessing an image dataset to building the Generator and Discriminator, training the GAN, and generating synthetic images.

The framework is designed to be domain-independent, allowing the same image-generation pipeline to be used with different types of image datasets, including medical, sports, manufacturing, and other domains.

The current test successfully loaded 4 images, completed the GAN training pipeline, and generated 20 synthetic images.

The next development stage is to connect the GAN module with the complete Streamlit frontend and test the system using larger real-world datasets.