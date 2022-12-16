import UrbanCtrl
import matplotlib as plt
import GlobalMapper
from PIL import Image
import os

"""
import *....
"""

params = {
    "pretrained_dir": '../UrbanCtrl/pretrained_model',
    "results_dir": '../UrbanCtrl/results',
}

a = GlobalMapper.GlobalMapper(params)
b = a.get_latent()
data = a.get_layout()
im = Image.fromarray(data)
im.save(os.path.join(a.results_dir, str(a.cur_idx) + "_a.jpeg"))


# # read a test reference image
# ref_img = plt.imread('xxxxx')
# new_layout = UrbanCtrl.template_based_layout(ref_img)

# plt.figure()
# plt.imshow(new_layout)
# plt.show()


# # tSNE_Poke
# start = [50, 50]
# end   = [100, 100]
# cur = 0.5
# poke_laytout = UrbanCtrl.tSNE_Poke(start, end, cur)

# plt.figure()
# plt.imshow(poke_laytout)
# plt.show()

# # ....




