import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from src.scripts.utils import *
from src.scripts.plotting import *
from src.scripts.distillation_trainer import DistillationTrainer
from src.scripts.preprocess import load_and_tokenize_data


def main():

    set_visible_gpu("2")
    data_folder = "src/data"
    
    teacher_model_name = "bert-base-uncased"
    logger.info("-------Load Tokenizer------------------")

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    
    tokenized_train, tokenized_test = load_and_tokenize_data(tokenizer, train_split="train[:1000]", test_split="test[:100]")
    
    num_labels = 2
    logger.info("-------Load Model------------------")

    teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name, num_labels=num_labels)
    
    training_args = TrainingArguments(
        output_dir="./teacher_results",
        evaluation_strategy="steps",
        eval_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model=teacher_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    teacher_eval_results = trainer.evaluate()
    print("Teacher Evaluation Results:", teacher_eval_results)
    
    teacher_predictions = trainer.predict(tokenized_test)
    teacher_logits = teacher_predictions.predictions
    teacher_pred_labels = np.argmax(teacher_logits, axis=-1)
    teacher_actual_labels = teacher_predictions.label_ids
    teacher_accuracy = accuracy_score(teacher_actual_labels, teacher_pred_labels)
    print("Teacher Model Accuracy:", teacher_accuracy)
    
    plot_predictions(teacher_actual_labels, teacher_pred_labels, 
                     "Teacher Model: Predictions vs. Actual Labels", f"{data_folder}/plots/teacher_model_predictions.png")
    plot_confusion_matrix(teacher_actual_labels, teacher_pred_labels, 
                          "Teacher Confusion Matrix", f"{data_folder}/plots/teacher_confusion_matrix.png")
    
    student_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
    
    distil_training_args = TrainingArguments(
        output_dir="./student_results",
        evaluation_strategy="steps",
        eval_steps=500,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    
    distil_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        args=distil_training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )
    
    distil_trainer.train()
    student_eval_results = distil_trainer.evaluate()
    print("Student (Distilled) Evaluation Results:", student_eval_results)
    
    student_predictions = distil_trainer.predict(tokenized_test)
    student_logits = student_predictions.predictions
    student_pred_labels = np.argmax(student_logits, axis=-1)
    student_actual_labels = student_predictions.label_ids
    student_accuracy = accuracy_score(student_actual_labels, student_pred_labels)
    print("Student Model Accuracy:", student_accuracy)
    
    plot_predictions(student_actual_labels, student_pred_labels, 
                     "Student Model: Predictions vs. Actual Labels", f"{data_folder}/plots/student_model_predictions.png")
    plot_confusion_matrix(student_actual_labels, student_pred_labels, 
                          "Student Confusion Matrix", f"{data_folder}/plots/student_confusion_matrix.png")
    
    plot_teacher_student_scatter(teacher_logits, student_logits, student_actual_labels, 
                                 f"{data_folder}/plots/teacher_vs_student_scatter.png")

if __name__ == "__main__":
    main()
