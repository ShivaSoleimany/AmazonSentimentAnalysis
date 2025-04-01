
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from loguru import logger

def plot_predictions(actual, predictions, title, filename, show_plot=False):
    logger.info("-------plot_predictions------------------")

    plt.figure(figsize=(8,6))
    plt.plot(range(len(actual)), actual, 'bo-', label="Actual")
    plt.plot(range(len(predictions)), predictions, 'ro-', label="Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Label")
    plt.title(title)
    plt.legend()
    if show_plot:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()

def plot_confusion_matrix(actual, predictions, title, filename):
    logger.info("-------plot_confusion_matrix------------------")

    cm = confusion_matrix(actual, predictions)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(filename)
    plt.close()

def plot_teacher_student_scatter(teacher_logits, student_logits, actual, filename):
    logger.info("-------plot_teacher_student_scatter------------------")

    # Compute positive class probabilities (assumes binary classification and index 1 is positive)
    teacher_probs = torch.softmax(torch.tensor(teacher_logits), dim=1).numpy()[:, 1]
    student_probs = torch.softmax(torch.tensor(student_logits), dim=1).numpy()[:, 1]
    
    plt.figure(figsize=(8,6))
    plt.scatter(teacher_probs, student_probs, c=actual, cmap="viridis", alpha=0.8)
    plt.xlabel("Teacher Positive Class Probability")
    plt.ylabel("Student Positive Class Probability")
    plt.title("Scatter Plot: Teacher vs. Student Predictions")
    plt.colorbar(label="Actual Label")
    plt.savefig(filename)
    plt.close()
