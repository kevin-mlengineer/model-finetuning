import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import logging
import sys
import os
import shutil
import tempfile

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure device for MacBook GPU
device = (
    "mps" 
    if torch.backends.mps.is_available() 
    else "cuda" 
    if torch.cuda.is_available() 
    else "cpu"
)
logger.info(f"Using device: {device}")

def load_and_prepare_data():
    """Load and tokenize AG News dataset."""
    try:
        logger.info("Loading AG News dataset...")
        # Create a temporary directory for dataset storage
        temp_dir = tempfile.mkdtemp()
        # Force re-download and avoid local caching issues
        ag_news_dataset = load_dataset(
            "ag_news",
            download_mode="force_redownload",
            cache_dir=temp_dir
        )
        train_data = ag_news_dataset['train']
        test_data = ag_news_dataset['test']

        logger.info("Loading tokenizer...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

        logger.info("Tokenizing dataset...")
        train_data = train_data.map(tokenize_function, batched=True)
        test_data = test_data.map(tokenize_function, batched=True)

        train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        return train_data, test_data, tokenizer
    except Exception as e:
        logger.error(f"Error in data loading: {str(e)}")
        raise

def compute_metrics(pred):
    """Compute evaluation metrics."""
    try:
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        return {"accuracy": acc, "f1": f1}
    except Exception as e:
        logger.error(f"Error in computing metrics: {str(e)}")
        raise

def train_baseline_student(train_data, test_data):
    """Fine-tune the baseline student model."""
    try:
        logger.info("Initializing baseline student model...")
        student_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)
        student_model.to(device)

        training_args = TrainingArguments(
            output_dir='./baseline_student',
            num_train_epochs=3,
            per_device_train_batch_size=32 if device == "mps" else 16,  # Increased batch size for MPS
            per_device_eval_batch_size=32 if device == "mps" else 16,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=2e-5,
            logging_dir='./logs_baseline',
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            use_mps_device=device == "mps",  # Enable MPS device
        )

        trainer = Trainer(
            model=student_model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            compute_metrics=compute_metrics,
        )

        logger.info("Training baseline student model...")
        trainer.train()
        logger.info("Evaluating baseline student model...")
        results = trainer.evaluate()
        return student_model, results['eval_accuracy'], results['eval_f1']
    except Exception as e:
        logger.error(f"Error in baseline training: {str(e)}")
        raise

class DistillationTrainer(Trainer):
    """Custom Trainer for Knowledge Distillation."""
    def __init__(self, teacher_model, temp=5.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temp = temp
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        try:
            outputs_student = model(**inputs)
            student_logits = outputs_student.logits
            student_loss = outputs_student.loss

            with torch.no_grad():
                outputs_teacher = self.teacher_model(**inputs)
                teacher_logits = outputs_teacher.logits

            loss_kl = nn.KLDivLoss(reduction="batchmean")
            student_logits_t = F.log_softmax(student_logits / self.temp, dim=-1)
            teacher_logits_t = F.softmax(teacher_logits / self.temp, dim=-1)
            distillation_loss = loss_kl(student_logits_t, teacher_logits_t) * (self.temp ** 2)

            total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
            return (total_loss, outputs_student) if return_outputs else total_loss
        except Exception as e:
            logger.error(f"Error in distillation loss computation: {str(e)}")
            raise

def train_distilled_student(train_data, test_data, teacher_model):
    """Train the student model with Knowledge Distillation."""
    try:
        logger.info("Initializing distilled student model...")
        student_model_distilled = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)
        student_model_distilled.to(device)
        teacher_model.to(device)

        distillation_args = TrainingArguments(
            output_dir='./distilled_student',
            num_train_epochs=3,
            per_device_train_batch_size=32 if device == "mps" else 16,  # Increased batch size for MPS
            per_device_eval_batch_size=32 if device == "mps" else 16,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=2e-5,
            logging_dir='./logs_distilled',
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            use_mps_device=device == "mps",  # Enable MPS device
        )

        distillation_trainer = DistillationTrainer(
            model=student_model_distilled,
            args=distillation_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            compute_metrics=compute_metrics,
            teacher_model=teacher_model,
            temp=5.0,
            alpha=0.5,
        )

        logger.info("Training distilled student model...")
        distillation_trainer.train()
        logger.info("Evaluating distilled student model...")
        results = distillation_trainer.evaluate()
        return student_model_distilled, results['eval_accuracy'], results['eval_f1']
    except Exception as e:
        logger.error(f"Error in distillation training: {str(e)}")
        raise

def evaluate_teacher(teacher_model, test_data):
    """Evaluate the teacher model."""
    try:
        logger.info("Evaluating teacher model...")
        teacher_model.to(device)
        teacher_trainer = Trainer(
            model=teacher_model,
            args=TrainingArguments(
                output_dir='./teacher_eval',
                per_device_eval_batch_size=32 if device == "mps" else 16,  # Increased batch size for MPS
                evaluation_strategy="no",
                do_train=False,
                use_mps_device=device == "mps",  # Enable MPS device
            ),
            eval_dataset=test_data,
            compute_metrics=compute_metrics,
        )
        results = teacher_trainer.evaluate()
        return results['eval_accuracy'], results['eval_f1']
    except Exception as e:
        logger.error(f"Error in teacher evaluation: {str(e)}")
        raise

def main():
    """Main function to execute the pipeline."""
    try:
        # Clear any existing cache in default location
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            logger.info("Cleared Hugging Face dataset cache.")

        # Load data and tokenizer
        train_data, test_data, tokenizer = load_and_prepare_data()

        # Load and freeze teacher model
        logger.info("Loading teacher model...")
        teacher_model = BertForSequenceClassification.from_pretrained('fabriceyhc/bert-base-uncased-ag_news', num_labels=4)
        teacher_model.to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        # Train and evaluate baseline student
        baseline_model, baseline_acc, baseline_f1 = train_baseline_student(train_data, test_data)

        # Train and evaluate distilled student
        distilled_model, distilled_acc, distilled_f1 = train_distilled_student(train_data, test_data, teacher_model)

        # Evaluate teacher
        teacher_acc, teacher_f1 = evaluate_teacher(teacher_model, test_data)

        # Compare parameter counts
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in baseline_model.parameters())

        # Print comparison table
        print("\nModel Comparison:")
        print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Parameters':<12}")
        print("-" * 52)
        print(f"{'Teacher (BERT)':<20} {teacher_acc:.4f}    {teacher_f1:.4f}    {teacher_params:,}")
        print(f"{'Baseline Student':<20} {baseline_acc:.4f}    {baseline_f1:.4f}    {student_params:,}")
        print(f"{'Distilled Student':<20} {distilled_acc:.4f}    {distilled_f1:.4f}    {student_params:,}")

        # Save models and tokenizer
        logger.info("Saving models and tokenizer...")
        baseline_model.save_pretrained('./baseline_student_model')
        distilled_model.save_pretrained('./distilled_student_model')
        tokenizer.save_pretrained('./tokenizer')

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()