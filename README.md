# Distill This! — Transformer Distillation Framework 🚀

![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-blue)
![Datasets](https://img.shields.io/badge/Data-HuggingFace%20Datasets-brightgreen)
![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-purple)
![Seaborn](https://img.shields.io/badge/Visualization-Seaborn-red)

Distill This! is a framework for performing teacher-student knowledge distillation using Hugging Face Transformers on a text classification task. In this project, we train a teacher model (BERT) on the [amazon_polarity](https://huggingface.co/datasets/amazon_polarity) dataset and then distill its knowledge into a smaller, more efficient student model (DistilBERT).

---

## Overview

Knowledge distillation is a technique where a larger, well-performing "teacher" model guides the training of a smaller "student" model. This project demonstrates:
- **Training a Teacher Model**: Fine-tuning a BERT model on a subset of the amazon_polarity dataset.
- **Distillation Process**: Transferring the learned knowledge from the teacher to a student model using a custom distillation loss.
- **Evaluation and Visualization**: Evaluating both models using accuracy metrics, confusion matrices, and visual comparisons of predictions.

---

## Features

- **Teacher-Student Distillation**: Learn how to train a smaller student model using a pre-trained teacher model.
- **Custom Evaluation Metrics**: Evaluate models using accuracy and confusion matrices.
- **Rich Visualizations**: Generate plots for predictions vs. actual labels, confusion matrices, and comparative scatter plots.
- **Modular and Configurable**: Easily adjust dataset sizes, training parameters, and GPU configurations.
- **Logging**: Integrated logging with [Loguru](https://github.com/Delgan/loguru) for detailed runtime information.

---

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Datasets](https://github.com/huggingface/datasets)
- [scikit-learn](https://scikit-learn.org/stable/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [loguru](https://github.com/Delgan/loguru)

You can install the required packages using:

```bash
pip install torch transformers datasets scikit-learn matplotlib seaborn loguru
```

## Usage

1- Clone the Repository:


```bash
git clone https://github.com/yourusername/distill-this.git
cd distill-this
```

2- Run the Training Script:

The main script trains the teacher model, performs distillation for the student model, and generates evaluation plots.

```bash
python -m src.app.main
```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.