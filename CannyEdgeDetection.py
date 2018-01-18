import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


loadimg = cv2.imread('b.jpg')
imgg = np.float32(loadimg)
img = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)

img2 = cv2.GaussianBlur(img,(5,5),0,0) 

kernelX = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
kernelY = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
magX = cv2.filter2D(img2,-1,kernelX)
magY = cv2.filter2D(img2,-1,kernelY)

direction = cv2.divide(magY,magX)

magX2 = cv2.multiply(magX,magX)
magY2 = cv2.multiply(magY,magY)
sum = cv2.add(magX2,magY2)
magnitude = cv2.sqrt(sum)

#cv2.imshow('image',magnitude)
#cv2.waitKey()

xSize = img.shape[0]
ySize = img.shape[1]

print(xSize, ySize)
edgeImage = np.zeros((xSize,ySize))

upperThreshold = 10

for row in range(xSize):
    for col in range(ySize):
        currDirection = math.atan(direction[row][col]) * 180/3.142

        while(currDirection<0):
            currDirection+=180;
        direction[row][col] = currDirection;

        if(magnitude[row][col] < upperThreshold) :
            continue

        isEdge = True

        if(currDirection>112.5 and currDirection <=157.5):
        
            if(col > 0 and row < ySize-1 and magnitude[row][col] <= magnitude[row+1][col-1] ) :
                isEdge = False;
            if(col < xSize-1 and row > 0 and magnitude[row][col] <= magnitude[row-1][col+1] ) :
                isEdge = False;
        
        elif(currDirection>67.5 and currDirection <= 112.5) :
        
            if(col > 0 and magnitude[row][col] <= magnitude[row][col-1] ) :
                isEdge = False;
            if(col < xSize-1 and magnitude[row][col] <= magnitude[row][col+1] ) :
                isEdge = False;
        
        elif(currDirection > 22.5 and currDirection <= 67.5) :
        
            if(col > 0 and row > 0 and magnitude[row][col] <= magnitude[row-1][col-1] ) :
                isEdge = False;
            if(col < xSize-1 and row < ySize-1 and magnitude[row][col] <= magnitude[row+1][col+1] ) :
                isEdge = False;
        
        else :
        
            if(row > 0 and magnitude[row][col] <= magnitude[row-1][col]) :
                isEdge = False;
            if(row < xSize-1 and magnitude[row][col] <= magnitude[row+1][col] ) :
                isEdge = False;
        

        if(isEdge):
            edgeImage[row][col] = 255

cv2.imshow('image',edgeImage)
cv2.waitKey()





'''plt.subplot(121),plt.imshow(magX),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magY),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()''

cv2.imshow('image',img)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()'''
#OpenCV uses BGR as its default colour order for images, matplotlib uses RGB
#a = cv2.cvtColor(magX, cv2.COLOR_BGR2RGB)
#b = cv2.cvtColor(magY, cv2.COLOR_BGR2RGB)