"""
This file is the interface file between the UI code and logic
"""

def Initialize(params: dict):
    """
    Initialization for the backend
    :param params: A dictionary that has all the params that backend needs
    :return: True/False to describe if the initialization succeed.
    """
    pass


def get_tSNE():
    """
    Return the current tSNE img result
    :return: np.array: H x W x 3
    """
    raise NotImplementedError('Not implemented yet')


def get_layout_img():
    """
    Return the current layout img result
    :return: np.array: H x W x 3
    """
    raise NotImplementedError('Not implemented yet')


def tSNE_Poke(start: list, end: list, cur:list):
    """
    Given the start, end, cur xy pixel position, compute the layout given the current xy pixel position.
    :param start: [x, y]
    :param end: [x, y]
    :param cur: [x, y]
    :return: t-SNE image, np.array: H x W x 3
    """
    raise NotImplementedError('Not implemented yet')


def template_based_layout(template_img: np.array):
    """
    Given the template_img, compute the layout based on the template
    :param template_img: np.array: H x W x 3
    :return: the layout image, np.array: H x W x 3
    """
    raise NotImplementedError('Not implemented yet')
