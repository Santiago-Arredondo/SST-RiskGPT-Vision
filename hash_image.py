from PIL import Image
from hashlib import md5
import numpy as np

img = Image.open("images/piso_resbalozo.png").convert("RGB")
np_img = np.array(img)
print("HASH:", md5(np_img).hexdigest())
