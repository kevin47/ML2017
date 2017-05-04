#!/usr/bin/env python
from keras.utils import plot_model
from keras.models import load_model
import sys

model = load_model(sys.argv[1])
model.summary()
plot_model(model, sys.argv[2])
