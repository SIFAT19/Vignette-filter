import cv2
import numpy as np
def changeRadius(value):
    global radius
    radius=value
    #for changing focus of the musk
def changeFocus(scope):
    global value
    value =scope

img =cv2.imread('taseen.jpg',3)

rows,cols=img.shape[:2]
value=1
radius=160
mask=np.zeros((int(rows*(value*0.1+1)),int(cols*(value*0.1+1))))
#createbtarcbar
cv2.namedWindow('trackbars')
cv2.createTrackbar('radius','trackbars',160,600,changeRadius)
cv2.createTrackbar('Focus','trackbars',1,10,changeFocus)



#generate mask using gaussian kernel

while (True):
    kernel_x=cv2.getGaussianKernel(int(cols*(0.1*value+1)),radius)
    kernel_y=cv2.getGaussianKernel(int(rows*(0.1*value+1)),radius)
    kernel= kernel_y *kernel_x.T

#normalizing the kernel
    kernel=kernel/np.linalg.norm(kernel)

#generating a mask to image
    mask=255*kernel
    output=np.copy(img)
# applying the mask to each channel in the input image
    mask_imposed=mask[int(0.1*value*rows):,int(0.1*value*cols):]
    for i in range(3):
        output[:,:,i]=output[:,:,i] *mask_imposed
    cv2.imshow('sd',img)
    cv2.imshow('vignette',output)
    key=cv2.waitKey(50)
    if(key==ord('q')):
        break
    elif(key==ord('s')):
        cv2.imwrite('output.jpg',output)

