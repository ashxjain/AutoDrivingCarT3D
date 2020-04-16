import numpy as np

from PIL import Image, ImageOps

def center_crop_img(img, x, y, crop_size):
  max_x, max_y = img.size
  pad_left, pad_right, pad_bottom, pad_top = 0, 0, 0, 0
  start_x = x - int(crop_size/2)
  start_y = y - int(crop_size/2)
  end_x = x + int(crop_size/2)
  end_y = y + int(crop_size/2)
  if start_x < 0:
      pad_left = -start_x
      start_x = 0
  if end_x >= max_x:
      pad_right = end_x - max_x
  if start_y < 0:
      pad_top = -start_y
      start_y = 0
  if end_y >= max_y:
      pad_bottom = end_y - max_y
  padding = (int(pad_left), int(pad_top), int(pad_right), int(pad_bottom))
  new_img = ImageOps.expand(img, padding, fill=255)
  crop_img = new_img.crop((start_x, start_y, start_x+crop_size, start_y+crop_size))

  return crop_img

def rotate_img(img, angle):
  im1 = img.convert('RGBA')
  rot = im1.rotate(angle)
  # a white image same size as rotated image
  fff = Image.new('RGBA', rot.size, (255,)*4)
  # create a composite image using the alpha layer of rot as a mask
  return Image.composite(rot, fff, rot).convert('L')

def show_img(ax, img, title=""):
  np_img = np.asarray(img)/255
  np_img = np_img.astype(int)
  ax.imshow(np_img, cmap='gray', vmin=0, vmax=1)
  if title != "":
    ax.set_title(title)
