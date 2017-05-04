#!/usr/bin/env python
import sys
import pandas as pd
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

df = pd.read_csv(sys.argv[1])
x = list(range(len(df)))

host = host_subplot(111, axes_class = AA.Axes)
ax2 = host.twinx()

host.set_xlabel('Epoch')
host.set_ylabel('Accuracy', color = 'b')
ax2.set_ylabel('Loss', color = 'r')

host.set_ylim(0, 1)
ax2.set_ylim(0, 3)

host.plot(df['train_acc'], 'b-', label = 'Training Accuracy')
ax2.plot(df['train_loss'], 'r-', label = 'Training Loss')
host.plot(df['val_acc'], 'b--', label = 'Validation Accuracy')
ax2.plot(df['val_loss'], 'r--', label = 'Validation Loss')

host.legend()
host.set_title('CNN')

#plt.draw()
#plt.show()
plt.savefig(sys.argv[2])
