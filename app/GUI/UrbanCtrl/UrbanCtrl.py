"""
This file is the interface file between the UI code and logic
"""
import GlobalMapper

global_mapper = None

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
    return GlobalMapper.cur_latent.copy()
    raise NotImplementedError('Not implemented yet')


def get_layout_img():
    """
    Return the current layout img result
    :return: np.array: H x W x 3
    """
    return GlobalMapper.cur_laytout.copy()
    raise NotImplementedError('Not implemented yet')


def tSNE_Poke(start: list, end: list, fract: float):
    """
    Given the start, end, cur xy pixel position, compute the layout given the current xy pixel position.
    :param start: [x, y]
    :param end: [x, y]
    :param fract: 0~1 float
    :return: Layout image, np.array: H x W x 3
    """
    raise NotImplementedError('Not implemented yet')


def template_based_layout(template_img: np.array):
    """
    Given the template_img, compute the layout based on the template
    :param template_img: np.array: H x W x 3
    :return: the layout image, np.array: H x W x 3
    """
    raise NotImplementedError('Not implemented yet')
