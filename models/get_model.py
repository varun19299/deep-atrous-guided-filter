"""
Get model
"""
from models.guided_filter import DeepAtrousGuidedFilter


def model(args):
    return DeepAtrousGuidedFilter(args)
