"""
Backend Implementation
"""


"""
....
"""

def initialize_model(model_weight: str):
    # model = ....
    return model


class GlobalMapper:
    def __init__(self, params: dict):
        self.cur_model = None
        self.cur_latent = None
        self.cur_layout = None


    def get_latent(self):
        return self.cur_latent.copy()


    def get_layout(self):
        return self.cur_layout.copy()


