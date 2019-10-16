import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

coordinateFrame={} # Dict to get the co-ordinates of the Pockets
redCords = [0,0] # Get the co-ordinates of the red pieces
isRedSpotted = False # Check whether red is detected
font = cv2.FONT_HERSHEY_SIMPLEX
white_x = [] # White pieces y coordinates
white_y = [] # White pieces x coordinates
black_x = [] # Black pieces y coordinates 
black_y = [] # Black pieces x coordinates
imgXmin = 15
imgXmax = 1050  # parameters to crop the image
imgYmin = 400
imgYmax = 1550
frameNo = 1

#coordinateArray = np.zeros((23,2),dtype = int) # Final coordinate array
# structure --> coordinateArray = |white_x(9)|white_y(9)|
#                                 |red_x(1)  |red_y(1)  |
#                                 |black_x(9)|black_y(9)|

#           coordinatePockets =   |pocketLT_x|pocketLT_y|
#                                 |pocketLB_x|pocketLB_y|
#                                 |pocketRB_x|pocketRB_y|
#                                 |pocketRT_x|pocketRT_y|


# function to sort the coordinate array
def coord_sort(array,coordinatePockets):
    """ This function takes a numpy array as the input and sort it according to the
        first element of the 2-D array.

        input : 19x2 array with the format of |white_x(9) | white_y(9) |
                                              |red_x(1)   | red_y(1)   |
                                              |black_x(9) | black_y(9) |

        output: 23x2 array with the format of |sorted_white_x(9) | sorted_white_y(9) |
                                              |red_x(1)          | red_y(1)          |
                                              |sorted_black_x(9) | sorted_black_y(9) |
                                              |holes_x(4)        | holes_y(4)        |
        note: sorting is done w.r.t x cordinates and y coordinates are considered if x coordinates
        are equal.
        """

    lst  = []
    lstf = np.zeros((23,2))
    arraySize = len(array)
    for i in range(arraySize):
        lst.append([array[i][0],array[i][1]])
    lstf[:9,:] = sorted(lst[:9])
    lstf[9,:] = lst[9]
    lstf[10:19,:] = sorted(lst[10:])
    lstf[19:23,:] = coordinatePockets
    #print(lstf)
    return np.array(lstf)


# function to perform non-max suppression
def non_max_suppression_fast(X,Y, overlapThresh,w1,h1):

    # if there are no boxes, return an empty list
    #if len(boxes) == 0:
    #    return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    #if boxes.dtype.kind == "i":
     #   boxes = boxes.astype("float")
 
    # initialize the list of picked indexes	
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = X
    y1 = Y
    x2 = x1 + w1
    y2 = y1 + h1
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        
	# grab the last index in the indexes list and add the
	# index value to the list of picked indexes
        last = len(idxs)-1
        i = idxs[last]
        pick.append(i)
     
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
     
            # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
     
            # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
     
            # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                        np.where(overlap > overlapThresh)[0])))
 
# return only the bounding boxes that were picked using the
# integer data type
    return X[pick],Y[pick]


for frame in range(frameNo,frameNo+1):#(frameNo,frameNo+1):#(6,23):
    coordinateFrame={} # Dict to get the co-ordinates of the Pockets
    redCords = [0,0] # Get the co-ordinates of the red pieces
    isRedSpotted = False # Check whether red is detected
    white_x = [] # White pieces y coordinates
    white_y = [] # White pieces x coordinates
    black_x = [] # Black pieces y coordinates 
    black_y = [] # Black pieces x coordinates
    coordinateArray = np.zeros((19,2),dtype = int)
    coordinatePockets = np.zeros((4,2),dtype = int)
    path = "opencv_frame_"+str(frame)+".png"
    #path = 'D:\\EducationFinalProject\\Vision\\images\\opencv_frame_'+str(frame)+'.png'
    #path = 'D:\\EducationFinalProject\\Vision\\images\\newfFrame'+str(frame)+'.png'
    img = cv2.imread(path)#load the image check 12,17
##    print(img.shape)
    #img = img[60:1020,450:1470,:] #*******************************************************
    img = img[imgXmin:imgXmax,imgYmin:imgYmax,:]
##    b,g,r = cv2.split(img)
##    blurg = cv2.GaussianBlur(g,(1,1),0)# should be r
##    smooth = cv2.addWeighted(blurg,-1.8,r,4.0,0)# if 4th argument if positive it becomes whitish  cv2.addWeighted(blur,2.0,img,1.5,0)
##    blurW = cv2.GaussianBlur(img,(1,1),0)
##    smoothW = cv2.addWeighted(blurW,0.7,img,0.5,0)
##    gray2 = cv2.cvtColor(smoothW,cv2.COLOR_BGR2GRAY) # converet to gray
##    #gray1 = smooth
##    gray = cv2.Canny(gray2, 80, 100)
    blur = cv2.GaussianBlur(img,(5,5),0)
    cv2.imshow('blur: ',imutils.resize(blur,width=700))
    #smooth = cv2.addWeighted(blur,2.5150,img,-1.505,0),cv2.addWeighted(blur,3.95150,img,0.4605,0)
    smooth = cv2.addWeighted(blur,1.95150,img,0.4605,0)
    gray = cv2.cvtColor(smooth,cv2.COLOR_BGR2GRAY)
    edgeImg = cv2.Canny(gray, 80, 15)

    # ---------- Pocket handdling ----------------

    # For Left Top Pocket
    tLefTop = cv2.imread("newTempLTP.jpg")
    blurLT = cv2.GaussianBlur(tLefTop,(1,1),0)
    smoothLT = cv2.addWeighted(blurLT,3.0,tLefTop,-2.5,0)
    tempLefTop = cv2.cvtColor(smoothLT,cv2.COLOR_BGR2GRAY)
    tempLefTop  = cv2.Canny(tempLefTop , 5, 20)

    w_lt,h_lt = tempLefTop.shape[::-1]
    
    #cv2.imshow('imgp1: ',imutils.resize(img,width=700))
    #cv2.imshow('smoothp: ',smoothLT)
    #cv2.imshow('grayp: ',tempLefTop)
    
    resultLT = cv2.matchTemplate(smooth,smoothLT,cv2.TM_CCOEFF_NORMED) # template matching on White Piece
    locLT = np.where(resultLT >= 0.59)
    (Xlt,Ylt) = locLT
    (Xlt,Ylt) = non_max_suppression_fast(Xlt,Ylt,0.2,w_lt,h_lt)
    print("len: ",len(locLT))

    for pt in range(len(Xlt)):
        x_i = Ylt[pt]+int(w_lt/2)-1
        y_i = Xlt[pt]+int(h_lt/2)-1
        #cv2.rectangle(img,(Ylt[pt],Xlt[pt]),(Ylt[pt]+w_lt,Xlt[pt]+h_lt),(255,0,0),3)
        cv2.circle(img,(x_i,y_i), 5, (255,0,255), -1)
        #coordinateFrame['LTP']=[y_i,x_i]
        coordinateFrame['LTP']=[x_i,y_i]
        print("coordinateFrame['LTP']: ",[x_i,y_i])
        #break
        
    # For Left Bottom Pocket
    tLefBottom = cv2.imread('newTempLBP1.jpg')
    blurLB = cv2.GaussianBlur(tLefBottom,(1,1),0)
    smoothLB = cv2.addWeighted(blurLB,3.0,tLefBottom,-2.5,0)
    tempLefTBottom = cv2.cvtColor(blurLB,cv2.COLOR_BGR2GRAY)
    tempLefTBottom  = cv2.Canny(tempLefTBottom , 5,20)

    w_lb,h_lb = tempLefTBottom.shape[::-1]

    resultLB= cv2.matchTemplate(smooth,smoothLB,cv2.TM_CCOEFF_NORMED) # template matching on White Piece
    locLB = np.where(resultLB >= 0.59)
    (Xlb,Ylb) = locLB
    (Xlb,Ylb) = non_max_suppression_fast(Xlb,Ylb,0.2,w_lb,h_lb)
    print("len: ",len(locLB))

    for pt in range(len(Xlb)):
        x_i = Ylb[pt]+int(w_lb/2)-1
        y_i = Xlb[pt]+int(h_lb/2)-1
        #cv2.rectangle(img,(Ylb[pt],Xlb[pt]),(Ylb[pt]+w_lb,Xlb[pt]+h_lb),(255,0,0),3)
        cv2.circle(img,(x_i,y_i), 5, (0,255,0), -1)
        #coordinateFrame['LBP']=[y_i,x_i]
        coordinateFrame['LBP']=[x_i,y_i]
        print("coordinateFrame['LBP']: ",[x_i,y_i])
        #break

    # For Right Bottom Pocket
    
    tRightBottom = cv2.imread('newTempRBP1.jpg')
    blurRB = cv2.GaussianBlur(tRightBottom,(1,1),0)
    smoothRB = cv2.addWeighted(blurRB,3.0,tRightBottom,-2.5,0)
    tempRightBottom = cv2.cvtColor(blurRB,cv2.COLOR_BGR2GRAY)
    tempRightBottom  = cv2.Canny(tempRightBottom , 5,20)

    w_rb,h_rb = tempRightBottom.shape[::-1]

    resultRB= cv2.matchTemplate(smooth,smoothRB,cv2.TM_CCOEFF_NORMED) # template matching on White Piece
    locRB = np.where(resultRB >= 0.59)
    (Xrb,Yrb) = locRB
    (Xrb,Yrb) = non_max_suppression_fast(Xrb,Yrb,0.2,w_lb,h_lb)
    #print("lenRb",print(locRB))

    for pt in range(len(Xrb)):
        x_i = Yrb[pt]+int(w_rb/2)-1
        y_i = Xrb[pt]+int(h_rb/2)-1
        #cv2.rectangle(img,(Yrb[pt],Xrb[pt]),(Yrb[pt]+w_rb,Xrb[pt]+h_rb),(255,0,0),3)
        cv2.circle(img,(x_i,y_i), 5, (225,0,0), -1)
        #coordinateFrame['RBP']=[y_i,x_i]
        coordinateFrame['RBP']=[x_i,y_i]
        print("coordinateFrame['RBP']: ",[x_i,y_i])
        #break

    coordinateFrame['LTP'] = [(coordinateFrame['LTP'][0]+coordinateFrame['LBP'][0])//2,coordinateFrame['LTP'][1]]
    coordinateFrame['LBP'] = [coordinateFrame['LTP'][0],coordinateFrame['LBP'][1]]
    coordinateFrame['RBP'] = [coordinateFrame['RBP'][0],(coordinateFrame['LBP'][1]+coordinateFrame['RBP'][1])//2]
    coordinateFrame['LBP'] = [coordinateFrame['LBP'][0],coordinateFrame['RBP'][1]]
    coordinateFrame['RTP'] = [coordinateFrame['RBP'][0],coordinateFrame['LTP'][1]]

    w_p = (w_lt+w_lb+w_rb)//3
    h_p = (h_lt+h_lb+h_rb)//3

    # Draw bounding boxes for refined values
##    cv2.rectangle(img,(coordinateFrame['LTP'][0]+h_p//2,coordinateFrame['LTP'][1]-w_p//2),(coordinateFrame['LTP'][0]-h_p//2,coordinateFrame['LTP'][1]+w_p//2),(255,0,0),3)
##    cv2.rectangle(img,(coordinateFrame['LBP'][0]+h_p//2,coordinateFrame['LBP'][1]-w_p//2),(coordinateFrame['LBP'][0]-h_p//2,coordinateFrame['LBP'][1]+w_p//2),(0,255,0),3)
##    cv2.rectangle(img,(coordinateFrame['RTP'][0]+h_p//2,coordinateFrame['RTP'][1]-w_p//2),(coordinateFrame['RTP'][0]-h_p//2,coordinateFrame['RTP'][1]+w_p//2),(0,0,255),3)
##    cv2.rectangle(img,(coordinateFrame['RBP'][0]+h_p//2,coordinateFrame['RBP'][1]-w_p//2),(coordinateFrame['RBP'][0]-h_p//2,coordinateFrame['RBP'][1]+w_p//2),(255,255,0),3)
    
    
    #----------------------- End of Pocket Handdling --------

    ## White pieces handdling
    templateWhite = cv2.imread('newTempWhite1.jpg')# white6
    blurW = cv2.GaussianBlur(templateWhite,(3,3),-1)
    smoothW = cv2.addWeighted(blurW,1.5,templateWhite,-0.1,0)
    templateGray = cv2.cvtColor(blurW,cv2.COLOR_BGR2GRAY)
    templateWhite = cv2.Canny(templateGray, 50, 100)

    # Template size extraction
    w_w,h_w = templateWhite.shape[::-1]

    # For white pieces
    resultWhite= cv2.matchTemplate(gray,templateWhite,cv2.TM_CCOEFF_NORMED)
    locWhite = np.where(resultWhite >= 0.27)#0.27
    (Xw,Yw) = locWhite
    #(Xwc,Ywc) = locWhite
    (Xwc,Ywc) = non_max_suppression_fast(Xw,Yw, 0.5,w_w,h_w)

    # Visuallization of the data
##    print("lenWhite: ",len(Xw))
    white_no = 0
    for pt in range(len(Xwc)):
        x_i = Ywc[pt]+int(w_w/2)
        y_i = Xwc[pt]+int(h_w/2)
        
        if ((img[y_i,x_i,0] >= 150 and img[y_i,x_i,1] >= 185) and img[y_i,x_i,2] >= 150):
##            print("white centers: ",pt+1," : ",img[y_i,x_i,:])
            s = np.sum(img[Ywc[pt]:Ywc[pt]+w_w,Xwc[pt]:Xwc[pt]+h_w,:])
            #cv2.rectangle(img,(Ywc[pt],Xwc[pt]),(Ywc[pt]+w_w,Xwc[pt]+h_w),(0,0,255),1)
            cv2.putText(img,"w"+str(white_no),(x_i,y_i), font, 0.5,(255,0,255),2,cv2.LINE_AA)
            #cv2.putText(img,str(pt+1),(Ywc[pt],Xwc[pt]), font, 2,(255,255,255),2,cv2.LINE_AA)
            white_y.append(y_i)
            white_x.append(x_i)
            white_no += 1
        

        if ((img[y_i,x_i,0] < 100 and img[y_i,x_i,1] < 100) and img[y_i,x_i,2] >= 190):
##            print("white centers: ",img[y_i,x_i,:])
            redCords.append(x_i)
            redCords.append(y_i)
            isRedSpotted = True
            s = np.sum(img[Ywc[pt]:Ywc[pt]+w_w,Xwc[pt]:Xwc[pt]+h_w,:])
            cv2.putText(img,"r"+str(1),(x_i,y_i), font, 0.5,(255,255,255),2,cv2.LINE_AA)
            #cv2.rectangle(img,(Ywc[pt],Xwc[pt]),(Ywc[pt]+w_w,Xwc[pt]+h_w),(0,255,0),1)
            

##    ## Red pieces handling
##    templateRed = cv2.imread('D:\\EducationFinalProject\\Vision\\templates\\red4.jpg')
##    bt,gt,rt = cv2.split(templateRed)
##    blurgt = cv2.GaussianBlur(gt,(3,3),0)
##
##    # Template size extraction
##    w_r, h_r = bt.shape[::-1]
##
##    # For red piece
##    if not(isRedSpotted):
##        resultRed= cv2.matchTemplate(blurg,blurgt,cv2.TM_CCOEFF_NORMED) # template matching on White Piece
##        locRed = np.where(resultRed >= 0.72)#0.27
##        (Xr,Yr) = locRed
##
##        # Data plotting for red
##        for pt in range(len(Xr)):
##            x_i = Yr[pt]+int(w_r/2)-1
##            y_i = Xr[pt]+int(h_r/2)-1
##            bb,gg,rr = img[y_i,x_i,:]
##            bb,gg,rr = int(bb),int(gg),int(rr)
##            if rr >=185 : # and bb <=90 :
##                cv2.rectangle(img,(Yr[pt],Xr[pt]),(Yr[pt]+w_r,Xr[pt]+h_r),(bb,gg,rr),3)
##                cv2.putText(img,"r"+str(1),(x_i,y_i), font, 0.5,(255,255,255),2,cv2.LINE_AA)
##                redCords.append(x_i)
##                redCords.append(y_i)
##                isRedSpotted = True
##                break
##    
##    

    ## Pockets Handdling

    # For Left Top Pocket
    tLefTop = cv2.imread('leftTopPocket.jpg')
    blurLT = cv2.GaussianBlur(tLefTop,(1,1),-1)
    smoothLT = cv2.addWeighted(blurLT,1.5,tLefTop,-0.1,0)
    tempLefTop = cv2.cvtColor(blurLT,cv2.COLOR_BGR2GRAY)
    tempLefTop  = cv2.Canny(tempLefTop , 90, 100)

    # For Left Bottom Pocket

    tLefBottom = cv2.imread('leftBottomPocket.jpg')
    blurLB = cv2.GaussianBlur(tLefBottom,(1,1),-1)
    smoothLB = cv2.addWeighted(blurLB,1.5,tLefBottom,-0.1,0)
    tempLefTBottom = cv2.cvtColor(blurLB,cv2.COLOR_BGR2GRAY)
    tempLefTBottom  = cv2.Canny(tempLefTBottom , 90, 100)

    # For Right Bottom Pocket

    tRightBottom = cv2.imread('rightBottomPocket.jpg')
    blurRB = cv2.GaussianBlur(tRightBottom,(1,1),-1)
    smoothRB = cv2.addWeighted(blurRB,1.5,tRightBottom,-0.1,0)
    tempRightBottom = cv2.cvtColor(blurRB,cv2.COLOR_BGR2GRAY)
    tempRightBottom  = cv2.Canny(tempRightBottom , 90, 100)

    # Take the dimensions of the pocket templates
    
    w_lt,h_lt = tempLefTop.shape[::-1]
    w_lb,h_lb = tempLefTBottom.shape[::-1]
    w_rb,h_rb = tempRightBottom.shape[::-1]


    
    coordinateFrame['LTP'] = [(coordinateFrame['LTP'][0]+coordinateFrame['LBP'][0])//2,coordinateFrame['LTP'][1]]
    coordinateFrame['LBP'] = [coordinateFrame['LTP'][0],coordinateFrame['LBP'][1]]
    coordinateFrame['RBP'] = [coordinateFrame['RBP'][0],(coordinateFrame['LBP'][1]+coordinateFrame['RBP'][1])//2]
    coordinateFrame['LBP'] = [coordinateFrame['LBP'][0],coordinateFrame['RBP'][1]]
    coordinateFrame['RTP'] = [coordinateFrame['RBP'][0],coordinateFrame['LTP'][1]]
##    cv2.circle(img,tuple(coordinateFrame['LTP']), 15, (0,255,255), -1)
##    cv2.circle(img,tuple(coordinateFrame['LBP']), 15, (0,255,255), -1)
##    cv2.circle(img,tuple(coordinateFrame['RBP']), 15, (0,255,255), -1)
##    cv2.circle(img,tuple(coordinateFrame['RTP']), 15, (0,255,255), -1)
##    print("coordinateFrame['LTP']: ",coordinateFrame['LTP'])
##    print("coordinateFrame['LBP']: ",coordinateFrame['LBP'])
##    print("coordinateFrame['RBP']: ",coordinateFrame['RBP'])
##    print("coordinateFrame['RTP']: ",coordinateFrame['RTP'])

    # Refine the pocket diamentions

    w_p = (w_lt+w_lb+w_rb)//3
    h_p = (h_lt+h_lb+h_rb)//3
    
##    print('LTP: ',coordinateFrame['LTP'])
##    print('LBP: ',coordinateFrame['LBP'])
##    print('RBP: ',coordinateFrame['RBP'])
##    print('RTP: ',coordinateFrame['RTP'])
##    print('w_p: ',w_p)
##    print('h_p: ',h_p)

    # Draw bounding boxes for refined values
    cv2.rectangle(img,(coordinateFrame['LTP'][0]+h_p//2,coordinateFrame['LTP'][1]-w_p//2),(coordinateFrame['LTP'][0]-h_p//2,coordinateFrame['LTP'][1]+w_p//2),(255,0,0),3)
    cv2.rectangle(img,(coordinateFrame['LBP'][0]+h_p//2,coordinateFrame['LBP'][1]-w_p//2),(coordinateFrame['LBP'][0]-h_p//2,coordinateFrame['LBP'][1]+w_p//2),(0,255,0),3)
    cv2.rectangle(img,(coordinateFrame['RTP'][0]+h_p//2,coordinateFrame['RTP'][1]-w_p//2),(coordinateFrame['RTP'][0]-h_p//2,coordinateFrame['RTP'][1]+w_p//2),(0,0,255),3)
    cv2.rectangle(img,(coordinateFrame['RBP'][0]+h_p//2,coordinateFrame['RBP'][1]-w_p//2),(coordinateFrame['RBP'][0]-h_p//2,coordinateFrame['RBP'][1]+w_p//2),(255,255,0),3)
##    cv2.circle(img,(coordinateFrame['RTP'][1],coordinateFrame['RTP'][0]), 5, (0,255,255), -1)
##    cv2.circle(img,(coordinateFrame['RBP'][1],coordinateFrame['RBP'][0]), 5, (0,255,255), -1)
##    cv2.circle(img,(coordinateFrame['LTP'][1],coordinateFrame['LTP'][0]), 5, (0,255,255), -1)
##    cv2.circle(img,(coordinateFrame['LBP'][1],coordinateFrame['LBP'][0]), 5, (0,255,255), -1)
    # For Black pieces
    templateBlack = cv2.imread('newTempBlack2.jpg')
    blurB = cv2.GaussianBlur(templateBlack,(1,1),0)
##    smootRB = cv2.addWeighted(blurB,4.550,templateBlack,-1.505,0)
    smootRB = cv2.addWeighted(blurB,0.95,templateBlack,0.6,0)
    tempSmoothRB = cv2.cvtColor(blurB,cv2.COLOR_BGR2GRAY)
    tempSmoothRB = cv2.Canny(tempSmoothRB , 30 , 80)#10 , 60

    w_b, h_b,_ = templateBlack.shape
    print(w_b, h_b)
    cv2.imshow('smoothi: ',imutils.resize(smooth,width=700))
    cv2.imshow('smoothRB: ',imutils.resize(edgeImg,width=700))
    cv2.imshow('Temp: ',smootRB)
    resultBlack = cv2.matchTemplate(smooth,smootRB,cv2.TM_CCOEFF_NORMED) # template matching on White Piece
    #resultBlack = cv2.matchTemplate(edgeImg,tempSmoothRB,cv2.TM_CCOEFF_NORMED)
    locBlack = np.where(resultBlack >= 0.528343)#0.56,24  0.13784352)
    (Xb,Yb) = locBlack
    (Xbc,Ybc) = non_max_suppression_fast(Xb,Yb, 0.2,w_b,h_b)
    
    black_no = 0
##    print('-----------------------------------------Black----------------------------------------------------------------------')
    for pt in range(len(Xbc)):
        x_i = Ybc[pt]+int(w_b/2)-1
        y_i = Xbc[pt]+int(h_b/2)-1
                            
        black_x.append(x_i)
        black_y.append(y_i)
        cv2.circle(img,(x_i,y_i),10,(0,255,0),-1)
        cv2.rectangle(img,(Ybc[pt],Xbc[pt]),(Ybc[pt]+w_b,Xbc[pt]+h_b),(0,255,0),1)
        cv2.putText(img,"b"+str(black_no),(x_i,y_i), font, 0.5,(255,255,255),2,cv2.LINE_AA)
            
##            if black_no == 0:
##                print("y_i: ",y_i," x_i: ",x_i)
##                print("LTP: ",coordinateFrame['LTP'])
##                print("LBP: ",coordinateFrame['LBP'])
##                print("RBP: ",coordinateFrame['RBP'])
        black_no += 1
##    print('black no. : ',black_no)

     
##    print('----------------------------------------EndBlack-------------------------------------------------------------------')

    # update the coordinateArray

    for i in range(9):
        if i < len(white_y):
            coordinateArray[i,0] = white_x[i]
            coordinateArray[i,1] = white_y[i]
        if i < len(black_y):
            coordinateArray[i+10,0] = black_x[i]
            coordinateArray[i+10,1] = black_y[i]

    coordinateArray[9,0],coordinateArray[9,1] = redCords[0],redCords[1]
    coordinatePockets[0,0],coordinatePockets[0,1] = coordinateFrame['LTP'][0],coordinateFrame['LTP'][1]
    coordinatePockets[1,0],coordinatePockets[1,1] = coordinateFrame['LBP'][0],coordinateFrame['LBP'][1]
    coordinatePockets[2,0],coordinatePockets[2,1] = coordinateFrame['RBP'][0],coordinateFrame['RBP'][1]
    coordinatePockets[3,0],coordinatePockets[3,1] = coordinateFrame['RTP'][0],coordinateFrame['RTP'][1]

    

    #### please return this values as well
    yNorm = max(coordinateFrame['LBP'][1],coordinateFrame['RBP'][1])
    xNorm = max(coordinateFrame['LBP'][0],coordinateFrame['RBP'][0])
##
    yMean = min(coordinateFrame['LTP'][1],coordinateFrame['RTP'][1])
    xMean = min(coordinateFrame['LBP'][0],coordinateFrame['LTP'][0])

    print("y min: ",yMean)
    print("y max: ",yNorm)


    sortedCordArray = coord_sort(coordinateArray,coordinatePockets)
    print(sortedCordArray)

    for i in range(19):    
        #print('xBlack, yBlack : ',int(sortedCordArray[i][1]*(yNorm-yMean) + yMean),int(sortedCordArray[i][0]*(xNorm-xMean) + xMean))
        cv2.circle(img,(int(sortedCordArray[i][0]),int(sortedCordArray[i][1])), 5, (255,0,255), -1)

    for j in range(19,23):
        cv2.circle(img,(int(sortedCordArray[j][0]),int(sortedCordArray[j][1])), 5, (255,0,255), -1)
    

    cv2.imshow('img: ',imutils.resize(img,width=700))
    cv2.waitKey(0)
    redCords=[]
    isRedSpotted = False
    
    cv2.imwrite(str(i)+'.png',img)
    #print("white_x ",white_x)
    #print("white_y ",white_y)
    white_x = []
    white_y = []
    black_x = []
    black_y = []
    cv2.destroyAllWindows()
    #cv2.imwrite(str(frame)+'.png',img)
##    print("Image: ",frame)
