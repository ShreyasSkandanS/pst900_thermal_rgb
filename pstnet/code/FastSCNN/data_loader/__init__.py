from .penn_dataset import fscnnSegmentation

datasets = {
    'fscnn': fscnnSegmentation,
}


def get_segmentation_dataset( **kwargs):
    """Segmentation Datasets"""
    return datasets["fscnn"](**kwargs)
