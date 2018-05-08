import cv2
import numpy as np
import scipy.stats as st

k = 0.04

#Load image
filename = '4'
img = cv2.imread(filename+'.jpg',0)
dimg = cv2.imread('4.jpg')

# Apply gaussian blur
res = img
img = cv2.GaussianBlur(res,(5,5),1,1) 
cv2.imshow('Input Image',img)
cv2.waitKey()

#Define the Horizontal & Vertical Filters
H = np.array([ [1,0,-1],[2,0,-2],[1,0,-1] ])
V = np.array([ [1,2,1],[0,0,0],[-1,-2,-1] ])

#Calculate Filter responses
horResponse = cv2.filter2D(img,-1, H )
verResponse = cv2.filter2D(img,-1, V )

xSize = img.shape[0]
ySize = img.shape[1]


Ix2 = np.multiply(horResponse, horResponse)
Iy2 = np.multiply(verResponse, verResponse)
Ixy = np.multiply(horResponse, verResponse)


Hscores = np.zeros((xSize,ySize))
offset = 2 #offset = windowSize/2

print('Starting to compute corners')
for x in range(offset,xSize-offset):
    for y in range (offset,ySize-offset):

        Ix2_matrix = Ix2[x-offset:x+offset+1,y-offset:y+offset+1]
        Ix2_mean = np.sum(Ix2_matrix)
       
        
        Iy2_matrix = Iy2[x-offset:x+offset+1,y-offset:y+offset+1]
        Iy2_mean = np.sum(Iy2_matrix)
        
        
        Ixy_matrix = Ixy[x-offset:x+offset+1,y-offset:y+offset+1]
        Ixy_mean = np.sum(Ixy_matrix)
        
        
        Matrix = np.array([ [Ix2_mean, Ixy_mean],[Ixy_mean, Iy2_mean] ])
        # Type cast to int because values may exceed 255
        determinant = (int)(Ix2_mean*Iy2_mean)-(int)(Ixy_mean*Ixy_mean)
        trace = (int)(Ix2_mean + Iy2_mean)
        R1 = (int)(determinant - k*(trace**2))
        Hscores[x][y] = R1
        
maxR = np.max(Hscores)
Threshold = abs(0.3 * maxR)

foundCorner = False
for y in range(xSize):
    for x in range(ySize):
        if(Hscores[y][x] > Threshold):
            foundCorner = True
            cv2.circle(dimg,(x,y),3,(0,0,255),-1)
            x = x + 1
    if(foundCorner):
        foundCorner = False
        y = y + 1

# Display the output
print('Done !')
cv2.imshow('My corners',dimg)
cv2.waitKey()

# Save the output 
output_file = filename+'_'+'MyCorners'+'.jpg'
cv2.imwrite(output_file,dimg)

