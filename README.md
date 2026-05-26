#  Neural Style Transfer using AdaIN

A Deep Learning-based Neural Style Transfer (NST) web application that transforms ordinary images into artistic creations by combining the content of one image with the style of another using **Adaptive Instance Normalization (AdaIN)** and **PyTorch**.

##  Live Demo

 **Try the application here:**  
https://asham-06-asha.hf.space

 **Hugging Face Space:**  
https://huggingface.co/spaces/Asham-06/Asha

---

##  Project Overview

Neural Style Transfer is a Computer Vision technique that applies the artistic style of one image to another while preserving its original content.

This project leverages:

- Adaptive Instance Normalization (AdaIN)
- Pretrained VGG Encoder
- Decoder Network for image reconstruction
- PyTorch for deep learning implementation
- Flask for backend integration
- Hugging Face Spaces for deployment

Users can upload a content image and a style image through a web interface and generate a stylized output image instantly.

---

##  Features

- Upload custom content images
- Upload custom style images
- Real-time image stylization
- Fast AdaIN-based style transfer
- User-friendly web interface
- Pretrained VGG feature extractor
- Public deployment using Hugging Face Spaces
- Download generated stylized images

---

##  Tech Stack

### Programming Language
- Python

### Deep Learning Framework
- PyTorch
- Torchvision

### Web Framework
- Flask

### Libraries
- NumPy
- Pillow

### Deployment
- Hugging Face Spaces

### Version Control
- Git
- GitHub

---

##  Project Structure

```text
nst_project/
│
├── content_data/
│   ├── avril.jpg
│   ├── brad_pitt.jpg
│   ├── flowers.jpg
│   └── ...
│
├── style_data/
│   ├── vangogh.jpg
│   ├── la_muse.jpg
│   ├── mondrian.jpg
│   └── ...
│
├── utils/
│   ├── models.py
│   └── utils.py
│
├── static/
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── experiment/
│   └── final_exp/
│
├── app.py
├── run_app.py
├── train.py
├── vgg_normalised.pth
└── README.md
```

---

##  Installation

### Clone the Repository

```bash
git clone https://github.com/Asha-m06/nst_project.git
cd nst_project
```

### Create Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install torch torchvision flask pillow numpy
```

---

##  Run Locally

Start the Flask application:

```bash
python run_app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5001/
```

---

##  Model Architecture

The workflow followed by the project:

```text
Content Image
      │
      ▼
VGG Encoder
      │
      ▼
Adaptive Instance Normalization (AdaIN)
      │
      ▼
Decoder Network
      │
      ▼
Stylized Output Image
```

### Pipeline

1. Upload Content Image
2. Upload Style Image
3. Extract Deep Features using VGG Encoder
4. Apply Adaptive Instance Normalization
5. Decode Modified Features
6. Generate Stylized Output Image

---

##  Sample Workflow

### Input

- Content Image
- Style Image

### Output

- Stylized Image preserving content structure and artistic style

Generated images are stored in:

```text
static/uploads/
```

---

##  Deployment

The application is deployed and publicly accessible through Hugging Face Spaces.

**Live Application:**  
https://asham-06-asha.hf.space

**Hugging Face Space:**  
https://huggingface.co/spaces/Asham-06/Asha

---

##  Applications

- Digital Art Generation
- Creative Image Editing
- Artistic Photo Transformation
- Deep Learning Demonstrations
- Computer Vision Research
- Educational Projects

---

##  Future Enhancements

- Multiple style blending
- Adjustable style strength
- Batch image processing
- Higher-resolution outputs
- Docker deployment
- GPU optimization
- Mobile-friendly interface

---

##  Author

**Asha M**

Information Science Engineering Student

Interests:
- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
- Full-Stack AI Applications

GitHub: https://github.com/Asha-m06

---

##  License

This project is intended for educational, research, and learning purposes.
