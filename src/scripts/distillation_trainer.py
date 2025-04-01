import torch
from transformers import Trainer
from loguru import logger
from src.scripts.utils import distillation_loss

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.to(self.args.device)
        self.teacher_model.eval()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        loss = distillation_loss(student_logits, teacher_logits, inputs["labels"])
        return (loss, student_outputs) if return_outputs else loss
