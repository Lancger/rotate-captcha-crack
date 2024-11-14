import argparse

import matplotlib.pyplot as plt
import torch
from PIL import Image

from rotate_captcha_crack.common import device
from rotate_captcha_crack.const import DEFAULT_CLS_NUM
from rotate_captcha_crack.model import RotNetR, WhereIsMyModel
from rotate_captcha_crack.utils import process_captcha

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", "-i", type=int, default=-1, help="Use which index")
    opts = parser.parse_args()

    with torch.no_grad():
        cls_num = DEFAULT_CLS_NUM
        model = RotNetR(cls_num=cls_num, train=False)
        model_path = WhereIsMyModel(model).with_index(opts.index).model_dir / "best.pth"
        print(f"Use model: {model_path}")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model = model.to(device=device)
        model.eval()

        img = Image.open("datasets/tieba/1615096444.jpg")
        img_ts = process_captcha(img, target_size=384)
        img_ts = img_ts.to(device=device)

        predict = model.predict(img_ts)
        degree = predict * 360
        print(f"Predict degree: {degree:.4f}°")

    img = img.rotate(
        -degree, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255)
    )  # use neg degree to recover the img
    plt.figure("debug")
    plt.imshow(img)
    plt.show()
