# Neural Style Transfer using AdaIN

A Deep Learning based Neural Style Transfer (NST) project that combines the content of one image with the artistic style of another image using Adaptive Instance Normalization (AdaIN) and a pretrained VGG network.

The project provides both training and inference capabilities along with a Flask-based web application for generating stylized images.

---

## Features

- Neural Style Transfer using AdaIN
- Fast image stylization
- Multiple content and style image support
- Pretrained VGG feature extractor
- Flask web interface
- Real-time image upload and stylization
- PyTorch implementation

---

## Project Structure

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

## Technologies Used

- Python
- PyTorch
- Torchvision
- Flask
- Pillow
- NumPy
- HTML
- CSS

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Asha-m06/nst_project.git
cd nst_project
```

### Create Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate environment:

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install torch torchvision flask pillow numpy
```

---

## Running the Application

Start the Flask server:

```bash
python run_app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5001/
```

---

## Training the Model

To train the Neural Style Transfer model:

```bash
python train.py
```

Training checkpoints and outputs will be saved inside the experiment directory.

---

## How Neural Style Transfer Works

1. Input a Content Image
2. Input a Style Image
3. Extract feature representations using a pretrained VGG network
4. Apply Adaptive Instance Normalization (AdaIN)
5. Decode transformed features
6. Generate the stylized output image

---

## Workflow

```text
Content Image
       │
       ▼
Pretrained VGG Encoder
       │
       ▼
AdaIN Layer
       │
       ▼
Decoder Network
       │
       ▼
Stylized Output Image
```

---

## Sample Images

### Content Images

Examples are available in:

```text
content_data/
```

### Style Images

Examples are available in:

```text
style_data/
```

### Output Images

Generated outputs are stored in:

```text
static/uploads/
```

---

## Example Usage

1. Launch the application

```bash
python run_app.py
```

2. Open

```text
http://127.0.0.1:5001/
```

3. Upload a content image
4. Upload a style image
5. Generate stylized artwork
6. Download the generated image

---

## Repository

GitHub Repository:

https://github.com/Asha-m06/nst_project

---

## Future Improvements

- Multiple style blending
- Adjustable style intensity
- High-resolution image generation
- Docker deployment
- GPU optimization
- Cloud deployment

---

## Author

**Asha M**

Information Science Engineering Student

Deep Learning | Computer Vision | Artificial Intelligence

---

## License

This project is developed for educational and research purposes.
