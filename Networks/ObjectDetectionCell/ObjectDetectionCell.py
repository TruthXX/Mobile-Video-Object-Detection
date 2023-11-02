import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from MobileNet import MobileNetV1,MobileNetV2
from SSD import SSD

class ObjectDetectionCell(nn.Module):
	"""
		Module to join encoder and decoder of predictor model
	"""
	def __init__(self, num_classes, width_mult):
		"""
		Arguments:
			pred_enc : an object of MobilenetV1 class
			pred_dec : an object of SSD class
		"""
		super(ObjectDetectionCell, self).__init__()

		self.pred_encoder = MobileNetV1(alpha = width_mult)
		self.pred_decoder = SSD(num_classes=num_classes, alpha = width_mult, is_test=False)
		

	def forward(self, seq):
		"""
		Arguments:
			seq : a tensor used as input to the model  
		Returns:
			confidences and locations of predictions made by model
		"""
		x = self.pred_encoder(seq)
		confidences, locations = self.pred_decoder(x)
		return confidences , locations

