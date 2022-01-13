import os
from transformers import AutoTokenizer, PretrainedConfig
import torch
import neopytorch
import numpy as np
import pickle
import textwrap
import logging

from sagemaker_inference import decoder

DEFAULT_MODEL_FILENAME = "model.pt"
DEFAULT_NEO_MODEL_FILENAME = "compiled.pt"

model_id2label = {"0": "NEGATIVE", "1": "POSITIVE"}

import subprocess

MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'


def model_fn(model_dir):
    
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        
        """Loads a model. For PyTorch, a default function to load a model only if Elastic Inference is used.
        In other cases, users should provide customized model_fn() in script.

        Args:
            model_dir: a directory where model is saved.

        Returns: A PyTorch model.
        """
        PATH_TO_NEO_MODEL = os.path.join(model_dir, DEFAULT_NEO_MODEL_FILENAME)
        PATH_TO_MODEL = os.path.join(model_dir, DEFAULT_MODEL_FILENAME)

        if os.path.exists(PATH_TO_NEO_MODEL):
            model_path = PATH_TO_NEO_MODEL
        elif os.path.exists(PATH_TO_MODEL):
            model_path = PATH_TO_MODEL
        else:
            raise FileNotFoundError("Failed to load model with default model_fn: missing file {}."
                                    .format(DEFAULT_MODEL_NEO_FILENAME))

        neopytorch.config(model_dir=model_dir, neo_runtime=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.jit.load(model_path, map_location=device)
        model = model.to(device)
            
        # We are doing the 1st inference here if device is cuda to build Tensorrt 
        # before running predict function as it takes time to build.
        sample_input_path = os.path.join(model_dir, 'sample_input.pkl')
        try:
            with open(os.path.join(model_dir, 'sample_input.pkl'), 'rb') as input_file:
                model_input = pickle.load(input_file)
            if torch.is_tensor(model_input):
                model_input = model_input.to(device)
                model(model_input)
            elif isinstance(model_input, tuple):
                model_input = (inp.to(device) for inp in model_input if torch.is_tensor(inp))
                model(*model_input)
            else:
                logging.warning(f'Only supports torch tensor or tuple of torch tensor. Input Type recieved: {type(model_input)}')
        except Exception as e:
            logging.warning(f"Sample input file is not present: {e}")
        finally:
            print("done loading model....")
        return {"model": model, "tokenizer": tokenizer}


def input_fn(input_data, content_type):
    decoded_input_data = decoder.decode(input_data, content_type)
    return decoded_input_data.tolist()


def predict_fn(data, model):
    inputs = model["tokenizer"](
        data["inputs"][0], data["inputs"][1], return_tensors="pt", max_length=128, padding="max_length", truncation=True
    )
    inputs_t = torch.LongTensor(inputs["input_ids"]), torch.LongTensor(inputs["attention_mask"])

    with torch.no_grad():
        predictions = model["model"](*inputs_t)[0]
        outputs = predictions.cpu().numpy()

    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)
    return [{"label": model_id2label[str(item.argmax())], "score": item.max().item()} for item in scores]