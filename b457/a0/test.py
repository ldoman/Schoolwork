from PIL import Image
pil_im = Image.open('apple.jpg').convert('L')
pil_im.show()
