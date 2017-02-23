#!/usr/bin/env python2
from PIL import Image, ImageDraw

lena = Image.open('lena.png')
lenapix = lena.load()
mody = Image.open('lena_modified.png')
modypix = mody.load()
draw = ImageDraw.Draw(mody)

for i in range(lena.size[0]):
	for j in range(lena.size[1]):
		if lenapix[i, j] == modypix[i, j]:
			draw.point((i, j), (0, 0, 0, 0))
		else:
			draw.point((i, j), modypix[i, j])

mody.save('ans_two.png', 'PNG')
