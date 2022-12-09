import UrbanCtrl

"""
import *....
"""

params = {
    'model_weight': ....,
    '.....'
}

Initialize(params)

# read a test reference image
ref_img = plt.imread('xxxxx')
new_layout = UrbanCtrl.template_based_layout(ref_img)

plt.figure()
plt.imshow(new_layout)
plt.show()


# tSNE_Poke
start = [50, 50]
end   = [100, 100]
cur = 0.5
poke_laytout = UrbanCtrl.tSNE_Poke(start, end, cur)

plt.figure()
plt.imshow(poke_laytout)
plt.show()

# ....




