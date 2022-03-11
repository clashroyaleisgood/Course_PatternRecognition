import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

def showPic(pic):
    # 3, 32, 32
    # pic = np.swapaxes(pic, 0, 2) # X.T 效果
    # pic = np.swapaxes(pic, 0, 1)
    plt.figure(figsize=(2, 2))
    plt.imshow(pic)
    plt.show()

def plot(imgs, orig_img, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

def trans_image(pic):
    resized_imgs = [T.Resize(size=size)(pic) for size in (30, 50, 100)]

    plot(resized_imgs, pic)