# CV-FinalProject

**Final project for 6.8301 Advances in Computer Vision**

## Project Overview

This project aims to create an innovative image compression tool that leverages both computer vision and natural language processing techniques. The core idea is to compress images by generating an edge mask using the Canny edge detection algorithm and a descriptive caption using GPT-4. These two components are then used to train a multimodal diffusion model to recover the original image.

## Workflow

1. **Edge Detection**: Extract an edge mask from the input image using the Canny edge detection algorithm.
2. **Caption Generation**: Generate a descriptive caption for the image using GPT-4.
3. **Multimodal Diffusion Model**: Train a multimodal diffusion model that takes the edge mask and the caption as inputs to reconstruct the original image.

## Key Components

### 1. Edge Detection

We use the Canny edge detection algorithm to create an edge mask of the original image. The edge mask captures the essential structural information of the image.

### 2. Caption Generation

Using GPT-4, we generate a detailed caption that describes the content of the image. This caption provides additional context that aids in the reconstruction of the image.

### 3. Multimodal Diffusion Model

We train a multimodal diffusion model with the edge mask and the caption as inputs. The model learns to reconstruct the original image by combining the structural information from the edge mask and the contextual information from the caption.

## Installation

To run this project, you will need to install the following dependencies:

- Python 3.8 or higher
- PyTorch
- torchvision
- OpenAI GPT-4 API
- NumPy
- Pillow
- matplotlib
- CompressAI

You can install the required Python packages using pip:

```bash
pip install torch torchvision numpy pillow matplotlib
