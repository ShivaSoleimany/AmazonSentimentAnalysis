import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from loguru import logger

def set_visible_gpu(gpu_id="2"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def compute_metrics(eval_pred):
    logger.info("-------compute_metrics------------------")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


def distillation_loss(student_logits, teacher_logits, labels, temp=2.0, alpha=0.5):
    logger.info("-------distillation_loss------------------")

    # Soft targets using temperature scaling
    student_probs = F.softmax(student_logits / temp, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
    # Compute KL divergence loss and cross-entropy loss
    kd_loss = F.kl_div(torch.log(student_probs), teacher_probs, reduction="batchmean")
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * ce_loss + (1 - alpha) * (temp**2) * kd_loss