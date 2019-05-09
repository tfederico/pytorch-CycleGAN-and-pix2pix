'''Steps:
1) remove green background
2) draw silhouette
3) crop around the hand
4) make crop square
5) crop original image
6) scale both to 256x256
7) translate and scale joints positions
8) save results
	- original image
	- silhouette
	- "new" joints positions
	- crop info (?)
'''

### IMPORTS

import argparse
import pandas as pd
import numpy as np
import cv2 as cv

### VARIABLES

config_name = "config"
config_ext = ".csv"
image_ext = ".png"
params_name = {"height": "h", "width": "w", "crop_height": "c_h", "crop_width": "c_w"} # header of the config file
RED, GREEN, BLUE = (2, 1, 0)

### FUNCTIONS

def read_params(path):
	"""Reads the parameters from the configuration file

	Parameters:
	    path (str): The configuration file path

	Returns:
	    params: a dictionary containing the parameters
	"""

	data = pd.read_csv(path+"/"+config_name+config_ext, header=0, dtype=int)	
	return data.copy().to_dict()

def read_image(path):
	"""Reads the image from memory

	Parameters:
	    path (str): The path to the image

	Returns:
	    img: the image
	"""

	img = cv.imread(path, cv.IMREAD_UNCHANGED)
	return img

def draw_silhouette(image):
	"""Draws the silhouette of the hand removing the green background

	Parameters:
	    image (array): Matrix representing the image

	Returns:
	    silhouette: the matrix representing the silhouette of the hand
	"""

	#reds = image[:, :, RED]
	greens = image[:, :, GREEN]
	#blues = image[:, :, BLUE]
		
	mask = (greens < 35) | (np.amax(image, axis=2) != greens)
	
	silhouette = np.where(mask, 255, 0)
	silhouette = np.array([255 - s for s in silhouette]) # makes hand black and background white

	return silhouette

def remove_background(image, silhouette, height, width):
	"""Removes the green background from the image

	Parameters:
	    image (array): Matrix representing the original image
		silhouette (array): Matrix representing the silhouette of the image

	Returns:
	    no_back_image: the matrix representing the original image without green background
	"""

	no_back_image = image.copy()
	
	for i in range(height):
		for j in range(width):
			if (silhouette[i,j] != 0):
				no_back_image[i,j] = 255

	return no_back_image
	

def crop_margins(image, height, width):
	"""Removes the white background around the hand silhouette

	Parameters:
	    image (array): Matrix representing the original image

	Returns:
	    margins: coordinates indicating where to crop the image
	"""
	pix_col = 0
	top = (pix_col in image[0, :])
	bottom = (pix_col in image[-1, :])
	left = (pix_col in image[:, 0])
	right = (pix_col in image[:, -1])


	top_index = 0
	bottom_index = 0
	left_index = 0
	right_index = 0
	
	
	done = (top and bottom and left and right)

	while(not done):
		if(not top and top_index < height):
			top_index += 1
		if(not left and left_index < width):
			left_index +=1
		if(not right and right_index < (width - 1)):
			right_index +=1
		if(not bottom and bottom_index < (height - 1)):
			bottom_index +=1

		right = (pix_col in image[:, width - (right_index + 1)])
		bottom = (pix_col in image[height - (bottom_index + 1), :])
		top = (pix_col in image[top_index, :])
		left = (pix_col in image[:, left_index])

		done = (top and bottom and left and right)

	bottom_index = height - 1 - bottom_index
	right_index = width - 1 - right_index
	
	margins = {"top": top_index, "bottom": bottom_index, "left": left_index, "right": right_index}
	
	return margins



def add_vertical_padding(margins, top_padding, bottom_padding, h):
	"""Adds vertical padding to make the image squared

	Parameters:
	    margins (array): array with the current margins used to crop the image
		top_padding (int): padding to add at the top
		bottom_padding (int) padding to add at the bottom	
		h (int): original height of the image
	Returns:
	    margins: array with the new margins used to crop the image
	"""
	if((margins["top"] - top_padding) < 0): # add the remaining padding to the bottom
				remaining_padding = abs(margins["top"] - top_padding)
				margins["top"] = 0
				margins["bottom"] += bottom_padding + remaining_padding
	elif((margins["bottom"] + top_padding) >= h):
		remaining_padding = (margins["bottom"] + top_padding) - h
		margins["bottom"] = h - 1
		margins["top"] -= top_padding + remaining_padding + 1
	else:
		margins["top"] -= top_padding
		margins["bottom"] += bottom_padding
	return margins

def add_horizontal_padding(margins, left_padding, right_padding, w):
	"""Adds horizontal padding to make the image squared

	Parameters:
	    margins (array): array with the current margins used to crop the image
		left_padding (int): padding to add to the left
		right_padding (int) padding to add to the right	
		w (int): original width of the image
	Returns:
	    margins: array with the new margins used to crop the image
	"""
	if((margins["left"] - left_padding) < 0): # add the remaining padding to the right
				remaining_padding = abs(margins["left"] - left_padding)
				margins["left"] = 0
				margins["right"] += right_padding + remaining_padding
	elif((margins["right"] + right_padding) >= w):
		remaining_padding = (margins["right"] + right_padding) - w
		margins["right"] = w - 1
		margins["left"] -= left_padding + remaining_padding + 1
	else:
		margins["left"] -= left_padding
		margins["right"] += right_padding

	return margins

def make_it_square(margins, h, w, c_h, c_w):
	"""Makes the image a square crop around the hand

	Parameters:
	    margins (array): array with the current margins used to crop the image
		h (int): original height of the image
		w (int): original width of the image
		c_h (int): desired crop height
		c_w (int): desired crop width
	Returns:
	    margins: array with the new margins used to crop the image
	"""

	h_c_h = margins["bottom"] - margins["top"] # hypothetical crop height
	h_c_w = margins["right"] - margins["left"] # hypothetical crop width

	if((h_c_h > c_h or h_c_w > c_w)): # make it square by adding the background
		if(h_c_h > h_c_w): # add columns because height > width
			left_padding = (h_c_h - h_c_w)//2
			right_padding = h_c_h - h_c_w - left_padding
			margins = add_horizontal_padding(margins, left_padding, right_padding, w)

		else: # add rows
			top_padding = (h_c_w - h_c_h)//2
			bottom_padding = h_c_w - h_c_h - top_padding
			margins = add_vertical_padding(margins, top_padding, bottom_padding, h)
			

	else:
		if(h_c_w < c_w): # add columns
			left_padding = (c_w - h_c_w)//2
			right_padding = c_w - h_c_w - left_padding
			margins = add_horizontal_padding(margins, left_padding, right_padding, w)

		if(h_c_h < c_h): # add rows
			top_padding = (c_h - h_c_h)//2
			bottom_padding = c_h - h_c_h - top_padding
			margins = add_vertical_padding(margins, top_padding, bottom_padding, h)
	
	return margins


def scale(image, height, width):
	"""Makes the image a square crop around the hand

	Parameters:
	    image (array): image to rescale
		height (int): new height of the image
		width (int): new width of the image
	Returns:
	    new_image (array): rescaled image
	"""
	image = image.astype('float32')
	new_image = cv.resize(image, dsize=(height, width))
	return new_image
	

def save_image(path, name, image):
	"""Makes the image a square crop around the hand

	Parameters:
		path (str): path to the folder in which the image is saved
		name (str): name of the file
	    image (array): image to rescale
	Returns:
	    None
	"""
	cv.imwrite(path+"/"+name, image)

def save_parameters(path, name, margins, h_sc, v_sc):
	"""Makes the image a square crop around the hand

	Parameters:
		path (str): path to the folder in which the parameters are saved
		name (str): name of the file
	    margins (array): crop margins
		h_sc (float): horizontal scale factor
		v_sc (float): vertical scale factor
	Returns:
	    None
	"""
	params = margins.copy()
	params["horizontal_scaling_factor"] = h_sc
	params["vertical_scaling_factor"] = v_sc

	df = pd.DataFrame.from_dict(params.values())
	df = df.transpose()
	df.to_csv(path+"/"+name, header=list(params.keys()), index=False)
	

def preprocess_image(image_path, config_path, results_path, results_name):
	"""Preprocess the image according to the configuration file

	Parameters:
	    image_path (str): The file location of the image
	    config_path (str): The configuration file path
		config_path (str): The results folder path

	Returns:
		None
	"""
	
	params = read_params(config_path)

	width = params[params_name["width"]][0]
	height = params[params_name["height"]][0]
	crop_width = params[params_name["crop_width"]][0]
	crop_height = params[params_name["crop_height"]][0]

	img = read_image(image_path)
	sil = draw_silhouette(img)

	margins = crop_margins(sil, height, width)
	
	### BUG FIXED
	h_c_h = margins["bottom"]-margins["top"]
	h_c_w = margins["right"]-margins["left"]
	if(h_c_h != h_c_w and h_c_h <= width and h_c_w <= height): # checking if it is possible or makes sense to make a square crop
		margins = make_it_square(margins, height, width, crop_height, crop_width)

	cropped_sil = sil[margins["top"]:margins["bottom"], margins["left"]:margins["right"]]
	cropped_img = img[margins["top"]:margins["bottom"], margins["left"]:margins["right"]]
	
	img_no_back = remove_background(cropped_img, cropped_sil, cropped_sil.shape[0], cropped_sil.shape[1])

	scaled_img_no_back = scale(img_no_back, crop_height, crop_width)
	scaled_cropped_sil = scale(cropped_sil, crop_height, crop_width)

	vertical_scaling_factor = crop_height/cropped_sil.shape[0]
	horizontal_scaling_factor = crop_width/cropped_sil.shape[1]
	
	no_back_name = "resized_" + results_name + image_ext
	cropped_sil_name = "silhouette_" + results_name + image_ext

	save_image(results_path, no_back_name, scaled_img_no_back)
	save_image(results_path, cropped_sil_name, scaled_cropped_sil)

	file_name = results_name + ".csv"
	save_parameters(results_path, file_name, margins, vertical_scaling_factor, horizontal_scaling_factor)
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-ip", "--image_path", help="path to the image to preprocess")
	parser.add_argument("-cp", "--config_path", help="path to the config file")
	parser.add_argument("-rp", "--results_path", help="path to the results folder")
	parser.add_argument("-rn", "--results_name", help="name for the results files")
	args = parser.parse_args()
	preprocess_image(args.image_path, args.config_path, args.results_path, args.results_name)
