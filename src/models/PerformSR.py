import sys
from pathlib import Path
original_path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.append(original_path)
import numpy as np
import cv2
import src.models.model as ml





def performSR(ref,path,filename):
    srcnn = ml.srcnnModel()
    srcnn.load_weights(f'{original_path}/weights/srcnn.h5')

    # convert the image to YCrCb - (srcnn trained on Y channel)
    temp = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)
    
    # create image slice and normalize  
    Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255

    # perform super-resolution with srcnn
    pre = srcnn.predict(Y, batch_size=1)
   
    pre = srcnn.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    temp[6: -6, 6: -6, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(f"{path}/data/processed/response/{filename}", img)
    return f"{path}/data/processed/response/{filename}"