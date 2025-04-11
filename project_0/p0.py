import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppresses TensorFlow logging: 0 = all, 1 = warning, 2 = error, 3 = fatal
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppresses HuggingFace warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_CPP_PLUGIN"] = "ERROR"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Makes TensorFlow think no GPU is available

import warnings
warnings.filterwarnings("ignore")

####

from transformers import pipeline

print("\nSentiment Analysis:")
classifier = pipeline("sentiment-analysis")
print(classifier("I've been waiting for a HuggingFace course my whole life."))

print("\nZero Shot Classification:")
classifier = pipeline("zero-shot-classification")
print(classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
))

# 4 - Downloading ESM

# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")



