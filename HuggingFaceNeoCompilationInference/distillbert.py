from sagemaker_inference import decoder
import torch


def input_fn(input_data, content_type):
    inputs = decoder.decode(input_data, content_type)
    inputs_t = torch.LongTensor(inputs[0]), torch.LongTensor(inputs[1])
    return inputs_t


def predict_fn(data, model):
    return model(*data)
