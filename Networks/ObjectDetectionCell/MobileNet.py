
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging


from ConvLayers import InvertedResidual,conv_1x1_bn,conv_bn,conv_dw

class MobileNetV1(nn.Module):
	def __init__(self, alpha=1):
		""" torch.nn.module for mobilenetv1 upto conv12
		Arguments:
			num_classes : an int variable having value of total number of classes
			alpha : a float used as width multiplier for channels of model
		"""
		super(MobileNetV1, self).__init__()
		# upto conv 12
		self.model = nn.Sequential(
			conv_bn(3, 32*alpha, 2),
			conv_dw(32*alpha, 64*alpha, 1),
			conv_dw(64*alpha, 128*alpha, 2),
			#conv_dw(128*alpha, 128*alpha, 1),
			conv_dw(128*alpha, 256*alpha, 2),
			#conv_dw(256*alpha, 256*alpha, 1),
			conv_dw(256*alpha, 512*alpha, 2),
			#conv_dw(512*alpha, 512*alpha, 1),
			#conv_dw(512*alpha, 512*alpha, 1),
			#conv_dw(512*alpha, 512*alpha, 1),
			#conv_dw(512*alpha, 512*alpha, 1),
			#conv_dw(512*alpha, 512*alpha, 1),
			)
		logging.info("Initializing weights of base net")
		self._initialize_weights()
		#self.fc = nn.Linear(1024, num_classes)
	def _initialize_weights(self):
		"""
		Returns:
			initialized weights of the model
		"""
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			
	def forward(self, x):
		"""
		Arguments:
			x : a tensor which is used as input for the model
		Returns:
			a tensor which is output of the model 
		"""
		x = self.model(x)
		return x

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1., dropout_ratio=0.2,
                 use_batch_norm=True, onnx_compatible=False):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, onnx_compatible=onnx_compatible)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                else:
                    self.features.append(block(input_channel, output_channel, 1,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel,
                                         use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


