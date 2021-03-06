import cv2
import numpy as np
import math

# Load input
filename = '4'
loadimg = cv2.imread(filename+'.jpg',0)
img = np.float32(loadimg)

# Smoothen the image before edge detection
img2 = cv2.GaussianBlur(img,(3,3),1) 

# Define sobel operator
kernelX = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
kernelY = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

# Find magnitude and direction of edges
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

# Define thresholds 
upperThreshold = 60
lowerThreshold = 20

# Do non maximum suppression
for row in range(xSize):
    for col in range(ySize):
        currDirection = math.atan(direction[row][col]) * 180/3.142

        while(currDirection<0):
            currDirection+=180;
        direction[row][col] = currDirection;


        # Ignore if magnitude is less than upper threshold
        if(magnitude[row][col] < upperThreshold) :
            continue

        isEdge = True

        # Check magnitude of neighbouring pixels(depending on direction of edge) to detect edge pixels

        if(currDirection > 112.5 and currDirection <= 157.5):
        
            if(col > 0 and row < xSize-1 and magnitude[row][col] <= magnitude[row-1][col-1] ) :
                isEdge = False;
            if(col < ySize-1 and row > 0 and magnitude[row][col] <= magnitude[row+1][col+1] ) :
                isEdge = False;
        
        elif(currDirection > 67.5 and currDirection <= 112.5) :
        
            if(row > 0 and magnitude[row][col] <= magnitude[row-1][col] ) :
                isEdge = False;
            if(row < xSize-1 and magnitude[row][col] <= magnitude[row+1][col] ) :
                isEdge = False;
        
        elif(currDirection > 22.5 and currDirection <= 67.5) :
        
            if(col > 0 and row > 0 and magnitude[row][col] <= magnitude[row+1][col-1] ) :
                isEdge = False;
            if(col < ySize-1 and row < xSize-1 and magnitude[row][col] <= magnitude[row-1][col+1] ) :
                isEdge = False;
        
        else :
        
            if(col > 0 and magnitude[row][col] <= magnitude[row][col-1]) :
                isEdge = False;
            if(col < ySize-1 and magnitude[row][col] <= magnitude[row][col+1] ) :
                isEdge = False;
        

        if(isEdge):
            edgeImage[row][col] = 255

cv2.imshow('After Non Maximum Suppression',edgeImage)
cv2.waitKey()
output_file = filename + '_'+'edge.jpg'
cv2.imwrite(output_file,edgeImage)
# Do Hysteresis Thresholding
imageChanged = True
i = 0
print('Starting Hysterisis Thresholding ')

while(imageChanged):
    imageChanged = False

    for row in range(xSize):
        for col in range(ySize):
            # Check for boundary condition
            if(row < 2 or row >= xSize-2 or col < 2 or col >= ySize-2) : 
                continue
            currDirection = direction[row][col]

            if(edgeImage[row][col]==255):
                edgeImage[row][col]=100
                if(currDirection > 112.5 and currDirection <= 157.5) :
                    if(row < xSize-1 and col > 0) :
                        if(lowerThreshold <= magnitude[row-1][col-1] and
                        edgeImage[row-1][col-1]!= 100 and
                        direction[row-1][col-1] > 112.5 and
                        direction[row-1][col-1] <= 157.5 and
                        magnitude[row-1][col-1] >= magnitude[row-2][col-2] and
                        magnitude[row-1][col-1] >= magnitude[row][col] ) :
                            edgeImage[row-1][col-1] = 255
                            imageChanged = True

                    if(col < ySize-1 and row > 0) :
                        if(lowerThreshold <= magnitude[row+1][col+1] and
                        edgeImage[row+1][col+1] != 100 and
                        direction[row+1][col+1] > 112.5 and
                        direction[row+1][col+1] <= 157.5  and
                        magnitude[row+1][col+1] >= magnitude[row][col] and
                        magnitude[row+1][col+1] >= magnitude[row+2][col+2]) :
                            edgeImage[row+1][col+1] = 255
                            imageChanged = True

                elif(currDirection > 67.5 and currDirection <= 112.5) :
                    if(row > 0) :
                        if(lowerThreshold <= magnitude[row-1][col] and
                        edgeImage[row-1][col]!= 100 and
                        direction[row-1][col] > 67.5 and
                        direction[row-1][col] <= 112.5  and
                        magnitude[row-1][col] >= magnitude[row-2][col] and
                        magnitude[row-1][col] >= magnitude[row][col]):
                            edgeImage[row-1][col] = 255
                            imageChanged = True
                        
                    if(row < xSize-1) :
                        if(lowerThreshold <= magnitude[row+1][col] and
                        edgeImage[row+1][col]!= 100 and
                        direction[row+1][col] > 67.5 and
                        direction[row+1][col] <= 112.5 and
                        magnitude[row+1][col] >= magnitude[row][col-1] and
                        magnitude[row+1][col] >= magnitude[row+2][col+1]):
                            edgeImage[row+1][col] = 255
                            imageChanged = True
                        
                elif(currDirection > 22.5 and currDirection <= 67.5) :
                    if(col > 0 and row < ySize-1) :
                        if(lowerThreshold <= magnitude[row+1][col-1] and
                        edgeImage[row+1][col-1]!= 100 and
                        direction[row+1][col-1] > 22.5 and
                        direction[row+1][col-1]  <= 67.5 and
                        magnitude[row+1][col-1] >= magnitude[row+2][col-2] and
                        magnitude[row+1][col-1] >= magnitude[row][col] ) :
                            edgeImage[row+1][col-1] = 255
                            imageChanged = True
                       
                    if(col < ySize-1 and row > 0) :
                        if(lowerThreshold <= magnitude[row-1][col+1] and
                        edgeImage[row-1][col+1] != 100 and
                        direction[row-1][col+1] > 22.5 and
                        direction[row-1][col+1] <= 67.5 and
                        magnitude[row-1][col+1] >= magnitude[row-2][col+2] and
                        magnitude[row-1][col+1] >= magnitude[row][col]) :
                            edgeImage[row-1][col+1] = 255
                            imageChanged = True
                        
                else :
                    if( col > 0) :
                        if(lowerThreshold <= magnitude[row][col-1] and
                        edgeImage[row][col-1]!= 100 and
                        direction[row][col] < 22.5 or
                        direction[row][col-1] >= 157.5 and
                        magnitude[row][col-1] >= magnitude[row][col-2] and
                        magnitude[row][col-1] >= magnitude[row][col]) :
                            edgeImage[row][col-1] = 255
                            imageChanged = True
    
                    if(col < ySize-1) :
                        if(lowerThreshold <= magnitude[row][col+1] and
                        edgeImage[row][col+1] != 100 and
                        direction[row][col+1] < 22.5 or
                        direction[row][col+1] >= 157.5 and
                        magnitude[row][col+1] >= magnitude[row][col] and
                        magnitude[row][col+1] >= magnitude[row][col+2] ) :
                            edgeImage[row][col+1] = 255
                            imageChanged = True
                       
    print('Finished iteration %d'%(i))
    i = i + 1
                        
for row in range(xSize):
        for col in range(ySize):
            if edgeImage[row][col] == 100 :
                edgeImage[row][col] = 255           

# Display output              
cv2.imshow('After Hysteresis Thresholding',edgeImage)
cv2.waitKey()

# Save output
output_file = filename + '_'+'Myedge.jpg'
cv2.imwrite(output_file,edgeImage)

