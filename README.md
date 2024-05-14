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

## Usage

### Download Dataset

1. Download the MS COCO 330k dataset from Hugging Face or some subset of it.

### Create Dataset

1. Run the notebook `example-pipeline/create-dataset.ipynb` to create the dataset.

### Train Model

1. Run the notebook `example-pipeline/training.ipynb` to train a model.

### Workflow

1. **Preprocess the Image**: Resize the image and convert it to a tensor.
2. **Generate Edge Mask**: Use the Canny edge detection algorithm to create an edge mask.
3. **Generate Caption**: Use GPT-4 to generate a descriptive caption for the image.
4. **Train the Model**: Train the multimodal diffusion model using the edge mask and the caption.
5. **Reconstruct the Image**: Use the trained model to reconstruct the original image from the edge mask and the caption.

### Example

Here's a simple example to demonstrate the training loop:

```python
import torch
import numpy as np
from my_model import MyModel, apply_canny_edge_detector, apply_canny_edge_detector_rgb

# Initialize model, optimizer, and loaders
model = MyModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader = ...  # Your training DataLoader
valid_loader = ...  # Your validation DataLoader
config = {'epochs': 10}  # Example configuration
T = 1000  # Total timesteps

# Training Loop
for epoch in range(config['epochs']):
    print("Starting Epoch: ", epoch)
    model.train()
    train_loss = 0
    total_bits = 0
    total_pixels = 0

    for images, captions in train_loader:
        # Model inputs
        m_rgb = apply_canny_edge_detector_rgb(images)
        m = apply_canny_edge_detector(images)
        t = torch.randint(0, T, (1,)).item()

        # Forward and reverse diffusion processes
        noised_images = model.forward_pass(m, m_rgb, captions, t)
        recovered_images = model.reverse_pass(noised_images, m, t)

        # Calculate loss
        loss = model.diffusion_loss((noised_images - images), (noised_images - recovered_images))
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save checkpoint
    checkpoint_path = f'checkpoints/epoch_{epoch}_checkpoint.pth'
    torch.save(model.state_dict(), checkpoint_path)
```

## Installation

To run this project, you will need to install the dependencies listed in the `requirements.txt` file.

You can install the required Python packages using pip:

```bash
pip install -r requirements.txt
