import torch
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import logging
import sys
import os

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

def verify_dependencies():
    """Verify that required dependencies are installed."""
    try:
        import transformers
        import datasets
        import sklearn
        import torch
        logger.info("All dependencies are installed successfully.")
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        logger.error("Please install dependencies using: pip install datasets transformers torch scikit-learn")
        sys.exit(1)

def load_and_prepare_test_data():
    """Load and tokenize the AG News test dataset."""
    try:
        logger.info("Loading AG News test dataset...")
        # Use a local cache directory compatible with macOS
        cache_dir = os.path.expanduser("~/huggingface_cache/datasets")
        os.makedirs(cache_dir, exist_ok=True)
        ag_news_dataset = load_dataset(
            "ag_news",
            split="test",
            cache_dir=cache_dir
        )

        logger.info("Loading tokenizer...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

        logger.info("Tokenizing test dataset...")
        test_data = ag_news_dataset.map(tokenize_function, batched=True)
        test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        return test_data, tokenizer
    except Exception as e:
        logger.error(f"Error in test data loading: {str(e)}")
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

def evaluate_model(model, test_data, model_name):
    """Evaluate the given model on the test dataset."""
    try:
        logger.info(f"Evaluating {model_name}...")
        eval_args = TrainingArguments(
            output_dir=f'./eval_{model_name}',
            per_device_eval_batch_size=16,
            evaluation_strategy="no",
            do_train=False,
        )
        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=test_data,
            compute_metrics=compute_metrics,
        )
        results = trainer.evaluate()
        return results['eval_accuracy'], results['eval_f1']
    except Exception as e:
        logger.error(f"Error in evaluating {model_name}: {str(e)}")
        raise

def load_model(model_path, model_type, num_labels):
    """Load a pre-trained or fine-tuned model, with error handling for file existence."""
    try:
        if model_type == "teacher":
            return model_type.from_pretrained(model_path, num_labels=num_labels)
        elif os.path.exists(model_path):
            return model_type.from_pretrained(model_path, num_labels=num_labels)
        else:
            raise FileNotFoundError(f"Model directory {model_path} not found. Please ensure the model was saved during training.")
    except Exception as e:
        logger.error(f"Error loading {model_path}: {str(e)}")
        raise

def main():
    """Main function to evaluate and compare models."""
    try:
        # Verify dependencies
        verify_dependencies()

        # Load test data
        test_data, tokenizer = load_and_prepare_test_data()

        # Load teacher model
        logger.info("Loading teacher model...")
        teacher_model = load_model('fabriceyhc/bert-base-uncased-ag_news', BertForSequenceClassification, num_labels=4)
        teacher_model.eval()

        # Load fine-tuned student model
        logger.info("Loading fine-tuned student model...")
        student_model = load_model('./distilled_student_model', DistilBertForSequenceClassification, num_labels=4)

        # Evaluate both models
        teacher_acc, teacher_f1 = evaluate_model(teacher_model, test_data, "teacher")
        student_acc, student_f1 = evaluate_model(student_model, test_data, "student")

        # Compare parameter counts
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())

        # Print comparison table
        print("\nModel Comparison on Test Data:")
        print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Parameters':<12}")
        print("-" * 52)
        print(f"{'Teacher (BERT)':<20} {teacher_acc:.4f}    {teacher_f1:.4f}    {teacher_params:,}")
        print(f"{'Fine-Tuned Student':<20} {student_acc:.4f}    {student_f1:.4f}    {student_params:,}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()