import numpy as np
import time
import math
import imutils
import matplotlib.pyplot as plt
import cv2
import block1WithoutNormalizing as b1

####################     import image   ####################################################
frame = b1.frameNo
imgXmin = b1.imgXmin
imgXmax = b1.imgXmax # parameters to crop the image
imgYmin = b1.imgYmin
imgYmax = b1.imgYmax


####################           Basic functions               ###############################
start=time.time()
piece_diameter=15# this is the radius of a piece in the image
disk_diameter=25
threshold=5
R = 30
r = 23


piecesMatrix = np.asarray(b1.sortedCordArray)
#print("p.m: ",piecesMatrix)
piecesMatrix[18,:]=[0,0]
######### filter out the pocketed pieces(negative pieces)
pieces_matrix = np.empty((23,2),dtype=object)
for i in range(23):
    if (i < 19) and ((piecesMatrix[i][0] <= 0) or (piecesMatrix[i][1] <= 0)):
        pieces_matrix[i][0] = float('nan')
        pieces_matrix[i][1] = float('nan')
    else:
        pieces_matrix[i][0] = piecesMatrix[i][0]
        pieces_matrix[i][1] = piecesMatrix[i][1]
#print('pieces matrix: ',pieces_matrix)
path = "opencv_frame_"+str(frame)+".png"
img = cv2.imread(path)#load the image    
img = img[imgXmin:imgXmax,imgYmin:imgYmax,:]

for i in range(23):
    #print(i)
    if not(math.isnan(pieces_matrix[i][0])):
        cv2.circle(img,(int(pieces_matrix[i][0]),int(pieces_matrix[i][1])), 5, (0,0,255), -1)
#cv2.imshow('img: ',imutils.resize(img,width=700))
cv2.waitKey(0)
cv2.destroyAllWindows()
#pieces_matrix2= np.empty(())
holes_matrix = np.asanyarray(pieces_matrix[19:,:])

#holes_matrix=np.asanyarray([[122,93],[122,846],[880,846],[880,93]])
print('holes: ',holes_matrix)


def gradient(coord1,coord2):
    #coord1 and coord2 are 2 coordinate pairs(x,y) of dim 2.
    epsilon = 1e-10
    return (coord2[1]-coord1[1])/((coord2[0]-coord1[0])+epsilon)

def intercept(coord,grad):
    #coord is a coordinate of a piece and grad is the gradient of the line.
    return coord[1]-grad*coord[0]

def line_finder(case):
    #pieces_matrix is a 19x2 numpy array with the center coordinates of the pieces
    #holes_matrix is a 4x2 numpy array with th center coordinates of the holes
    if case=='Holes':
        gradients=np.zeros((19,4))
        intercepts=np.zeros((19,4))
        for i in range(19):
            for j in range(4):
                m=gradient(pieces_matrix[i,:],holes_matrix[j,:])
                gradients[i,j]=m
                intercepts[i,j]=intercept(pieces_matrix[i,:],m)
        return gradients,intercepts
    
    elif case=='Disk':
        num_pos=disk_positions.shape[0]
        gradients=np.zeros((19,num_pos))
        intercepts=np.zeros((19,num_pos))
        for i in range(19):
            for j in range(num_pos):
                m=gradient(pieces_matrix[i,:],disk_positions[j,:])
                gradients[i,j]=m
                intercepts[i,j]= intercept(pieces_matrix[i,:],m)
        return gradients,intercepts
        
        

def perpendicular_distance(coord,grad,intercept):
    return (np.absolute(coord[1]-grad*coord[0]-intercept)/np.sqrt(np.square(grad)+1))

def region_checker(gradients,intercepts,case):
    if case=='Holes':
        dist_matrix=np.zeros((19,4,19))
        for i in range(19):
            for j in range(4):
                for k in range(19):
                    dist_matrix[i,j,k]=perpendicular_distance(pieces_matrix[k,:],gradients[i,j],intercepts[i,j])
        return dist_matrix
    
    elif case=='Disk':
        num_pos=disk_positions.shape[0]
        dist_matrix=np.zeros((19,num_pos,19))
        for i in range(19):
            for j in range(num_pos):
                for k in range(19):
                    dist_matrix[i,j,k]=perpendicular_distance(pieces_matrix[k,:],gradients[i,j],intercepts[i,j])
        return dist_matrix

def angle_calculator(m1,m2):
    if m2*m1==-1:
        return 90.0
    else:
        return math.degrees(math.atan((m2-m1)/(1+m1*m2)))
    
def angle_error_correction(piece,disk_pos,hole):
    piece_loc = pieces_matrix[piece]
    disk_loc = disk_positions[disk_pos]
    hole_loc = holes_matrix[hole]
    alpha = math.atan(gradients[piece,hole]) #Radians
    if (piece_loc[0]<=disk_loc[0]):# and (piece_loc[1]>disk_loc[1]):
        new_x = piece_loc[0]+(R+r)*math.cos(alpha)
        new_y = piece_loc[1]+(R+r)*math.sin(alpha)
        new_coord = [new_x,new_y]
    
        
    else:
        new_x = piece_loc[0]-(R+r)*math.cos(alpha)
        new_y = piece_loc[1]-(R+r)*math.sin(alpha)
        new_coord = [new_x,new_y]
        
      
#    if (piece_loc[0]<disk_loc[0]) and (piece_loc[1]<disk_loc[1]):
#        new_x = piece_loc[0]+(R+r)*math.cos(alpha)
#        new_y = piece_loc[1]+(R+r)*math.sin(alpha)
#    
#        new_coord = [new_x,new_y]
#    
#        return math.degrees(math.atan(gradient(new_coord,disk_loc))),new_coord
    return math.degrees(math.atan(gradient(new_coord,disk_loc))),new_coord  
##def lineDrawer(x_bar,y_bar,m,c,i):
##    #print('')
##    x_cord = (y_bar+x_bar/m-c)/(m + 1/m)
##    y_cord = (c + ((m**2)*y_bar + m*x_bar)/(m**2 + 1))
##    #print(x_cord ,y_cord)
##    cv2.circle(img,(int(x_cord),int(y_cord)), 5, (10*i,30*i,255), -1)
##    cv2.circle(img,(int(x_bar),int(y_bar)), 5, (10*i,30*i,255), -1)
##    #cv2.line(img,(int(x_bar*dx + xMean),int(y_bar*dy + yMean)),(int(y_cord*dy + yMean),int(x_cord*dx + xMean)),(0,180,180),1,1)
##
##    cv2.circle(img,(int(y_bar*dy + yMean),int(x_bar*dx + xMean)), 5, (0,0,255), -1)
##    cv2.imshow('img: ',imutils.resize(img,width=700))
    #return [x_cord,y_cord]
########################################################################################        
    
#################################          Step 1               #######################
#This part is to,
# 1) find the lines connecting pieces and holes.
# 2) find the perpendicular distances to those lines from all the other pieces
gradients,intercepts=line_finder(case='Holes')
gradients,intercepts = gradients,intercepts#-1*gradients,-1*intercepts
distances=region_checker(gradients,intercepts,case='Holes')

#print(np.array((distances>(piece_diameter+threshold)),dtype=int))
print('time spent step 1-',time.time()-start,'s' )        
        
#################################          Step 2               #######################
#This part is to,
# 1) find the lines connecting pieces and disk positions.
# 2) find the perpendicular distances to those lines from all the other pieces
start2= time.time()
disk_positions = np.array([[930,930,930,930,930],[210,370,510,650,805]]).T
cv2.circle(img,(745,187), 2, (0,255,255), -1)
cv2.circle(img,(745,667), 2, (0,255,255), -1)
#cv2.imshow('imgCrl',img)
disk_x = [930,930,930,930,930] #[270,360,460,600,744]
disk_y = [210,370,510,650,805] #[195,195,195,195,195]

for i in range(len(disk_x)):
    cv2.circle(img,(disk_x[i],disk_y[i]), 5, (0,0,255), -1)
cv2.imshow('imgDisk: ',imutils.resize(img,width=700))


gradients2,intercepts2=line_finder(case='Disk')
gradients2,intercepts2 = gradients2,intercepts2#-1*gradients2,-1*intercepts2
distances2=region_checker(gradients2,intercepts2,case='Disk')

#print(np.array((distances2>(piece_diameter+threshold)),dtype=int))
print('time spent step 2-',time.time()-start2,'s' )  

#################################          Step 3               #######################
#This part is to calculate the angles between the straight lines connecting a piece to a hole and a disk position
start3=time.time()
ang_thresh=40
num_pieces=19
num_holes=4
num_pos= disk_positions.shape[0]
angles=np.zeros((num_pieces,num_holes,num_pos))
for i in range(num_pieces):
    #print('piece number: ',i)
    for j in range(num_holes):
        for k in range(num_pos):
            angles[i,j,k]=angle_calculator(gradients[i,j],gradients2[i,k])

#print(np.array((-ang_thresh<angles<ang_thresh)),dtype=int))
print('time spent step 3-',time.time()-start3,'s' )  

################################         Step 4                 ########################
#This part is to evaluate all the features in the previous steps and calculate a ranking score metric.
start4= time.time()
#print('thresh piece to hole: ',piece_diameter+threshold)
#print('distance: ',distances)
feature1=np.sum(np.array((distances>(piece_diameter+threshold)),dtype=int),axis=2)
feature2=np.sum(np.array((distances2>(disk_diameter+threshold+piece_diameter)),dtype=int),axis=2)
print('feature1:',feature1)
print('feature2:',feature2)
feature3=np.logical_and(np.array((angles<ang_thresh),dtype=int), np.array((-ang_thresh<angles)) ,dtype=int)#np.array((-ang_thresh<angles)
feature3=np.array(feature3,dtype=int)
#print(feature3)
score_matrix=np.zeros((num_pieces,num_holes,num_pos))
for i in range(num_pieces):
    for j in range(num_holes):
        for k in range(num_pos):
            score_matrix[i,j,k]= feature1[i,j]+feature2[i,k] #feature2[i,k]#feature1[i,j]#+feature2[i,k] #feature1[i,j]+feature2[i,k] ###-------------editing here
        
rank_matrix = np.multiply(score_matrix,feature3)#score_matrix#score_matrix #score_matrix#np.multiply(score_matrix,feature3)
print('rank mat: ',rank_matrix)

possible_shots=[]
maximum = np.max(rank_matrix)
print('maximum: ',maximum)
for i in range(19):
    for j in range(4):
        for k in range(5):
            #if rank_matrix[i][j][k]== maximum:
            if rank_matrix[i][j][k] == maximum:
                possible_shots.append([i,j,k])
#print(possible_shots)
print("Possible len: ",possible_shots)
##forward_possible = []
##backward_possible = []
##for shot in possible_shots:
##    t_piece = pieces_matrix[shot[0]]
##    t_hole = holes_matrix[shot[1]]
##    t_diskpos = disk_positions[shot[2]]
##    #print ('tpiece',t_piece[1])
##    #print('thole',t_hole[1])
##    #print(t_diskpos)
##    metric = (t_piece[1]-t_diskpos[1])*(t_hole[1]-t_piece[1])
##    if metric>=0:
##        forward_possible.append(shot)
##    else:
##        backward_possible.append(shot)
##    #print (t_piece)
##    #print(t_hole)
##    #print(t_diskpos)
##    
##
##
##possible_shots = forward_possible                
##yMax = b1.yNorm

##################### Seperating back shots and forward shots ################################

forward_possible = []
backward_possible = []
for shot in possible_shots:
    t_piece = pieces_matrix[shot[0]]
    t_hole = holes_matrix[shot[1]]
    t_diskpos = disk_positions[shot[2]]
    #print ('tpiece',t_piece[1])
    #print('thole',t_hole[1])
    #print(t_diskpos)
    metric1 = (t_piece[0]-t_diskpos[0])*(t_hole[0]-t_piece[0])
    metric2 = (t_piece[1]-t_diskpos[1])*(t_hole[1]-t_piece[1])
    if metric1>=0 and metric2>=0 :
        forward_possible.append(shot)
    else:
        backward_possible.append(shot)
    
    #print (t_piece)
    #print(t_hole)
    #print(t_diskpos)
    


#print("holes",(holes_matrix))
possible_shots = forward_possible

######################## Given the possible shot calculating the parameters of the shot #########################
def distance_calc(p1,p2):
    return np.sqrt(np.square(p1[0]-p2[0])+np.square(p1[1]-p2[1]))

m_S = 110           #Mass of the pendulum/shooter
m_D = 15            #Mass of the disk 
m_P = 5             #Mass of a carrom piece in g
d_D = 40            #Diamter of the disk in mm
v_Dmax = 3.4       #Maximum velocity the disk can have in m/s.
v_Pmax = 1          #Maximum velocity a piece can have right before a pocket without bouncing back  
diagonal = (distance_calc(holes_matrix[0],holes_matrix[2])+distance_calc(holes_matrix[1],holes_matrix[3]))/2 #diagonal distance
Friction_coeff = 0.01
g = 9.81            #gravitation acceleration in m/s.
e = 0.7             #elasticity coefficient
def params_calculation(shot):
    #shot is a 3 dimensional array which gives 1) target piece 2) target hole 3) disk position
    piece_loc = pieces_matrix[shot[0]]
    hole_loc = holes_matrix[shot[1]]
    diskpos_loc = disk_positions[shot[2]]
    dirn = piece_loc[0]-diskpos_loc[0]
    if dirn >=0:
        direction = 'back'
    else:
        direction = 'forward'
    
    dist_P = distance_calc(piece_loc,hole_loc)          #Distance the piece has to travel to reach the hole
    dist_D = distance_calc(diskpos_loc,piece_loc)       #Distance the disk has to travel to reach the piece
    acc = -Friction_coeff*g         #Deceleration due to friction
    
    v_P_init = np.sqrt((v_Pmax**2)-2*acc*dist_P) #Initial velocity the piece should have to reach the hole V2 =U2-2AS
    theta = math.radians(angles[shot[0],shot[1],shot[2]])
    tan_alpha = math.tan(theta)*(1+e)/(e-1)
    
    v_D_final = v_P_init*(math.cos(theta)+(math.sin(theta)/tan_alpha))/e
    
    v_D_init = np.sqrt((v_D_final**2)-2*acc*dist_D)
    
    return v_D_init,direction

print('possible shots:',possible_shots)
#shoot =[]
#shoot2 =[]
count=1
for s in possible_shots:
    disk_v,direction = params_calculation(s)
    #print('gradients:',gradients2)
    shooting_angle = math.degrees(math.atan(gradients2[s[0],s[2]]))
    corrected,new = angle_error_correction(s[0],s[2],s[1])
    
    cv2.line(img,(int(new[0]),int(new[1])),(int(disk_x[s[2]]),int(disk_y[s[2]])),(255,0,0),1,1)
    cv2.circle(img,(int(new[0]),int(new[1])),30,(0,255,0),-1)
    cv2.circle(img,(int(new[0]),int(new[1])),5,(0,0,255),-1)
    
    cv2.imshow('img: ',imutils.resize(img,width=700))
    cv2.waitKey(0)
    
    if direction=='back':
        if shooting_angle>=0:
            shooting_angle = -180 + shooting_angle
        else:
            shooting_angle = 180 + shooting_angle
#    shoot.append(gradients2[s[0],s[2]])
#    shoot2.append(intercepts2[s[0],s[2]])
    print('Disk velocity:'+str(count),disk_v)
    print('Shooting angle:'+str(count),shooting_angle)
    print('Corrected angle:',str(count),corrected)
    count+=1
##print('yMax: ',yMax)
for i,array in enumerate(possible_shots):
    path = "opencv_frame_"+str(frame)+".png"
    img = cv2.imread(path)#load the image
    
    img = img[imgXmin:imgXmax,imgYmin:imgYmax,:]
    piece, hole, pos = array[0],array[1],array[2]
    piece, hole, pos = int(piece),int(hole),int(pos)
    cv2.circle(img,(int(pieces_matrix[piece][0]),int(pieces_matrix[piece][1])), 5, (0,255,255), -1)
#    for j in range(imgXmin,imgXmax):
#        cv2.circle(img,(j,int(shoot[i]*j+shoot2[i])),5,(255,0,0),-1)
    
##    print(' x >>>>>>>>>>>>>>>>> correct : ',int(pieces_matrix[piece][0]*dx + xMean))
##    print(' y >>>>>>>>>>>>>>>>> correct : ',int(pieces_matrix[piece][1]*dy + yMean))
##    print(' y >>>>>>> pieces_matrix[piece][1] >>>>>>>>>> correct : ',pieces_matrix[piece][1])
##    print(' x >>>>>>> dx >>>>>>>>>> correct : ',dx)
##    print(' y >>>>>>> dy >>>>>>>>>> correct : ',dy)
##    print(' x >>>>>>> pieces_matrix[piece][0] >>>>>>>>> correct : ', pieces_matrix[piece][0])
    #for element in range(9):
    #    cv2.putText(img,"w"+str(element+1),(int(pieces_matrix[element][1]*dy + yMean),int(pieces_matrix[element][0]*dx + xMean)), font, 0.5,(0,0,0),2,cv2.LINE_AA)
    #for k in range(len(disk_x)):
##    for i in range(19):
##        if not(math.isnan(pieces_matrix[i][0])):
##            #cv2.line(img,(int(pieces_matrix[piece][1]),int(pieces_matrix[piece][0])),(int(disk_x[k]*dx + xMean),int(disk_y[k]*dy + yMean)),(0,0,0),1,1)
##            lineDrawer(pieces_matrix[i][0],pieces_matrix[i][1],gradients2[piece][pos],intercepts2[piece][pos],i)
##    print("True Grad: ", (disk_y[pos]-pieces_matrix[piece][1])/(disk_x[pos]-pieces_matrix[piece][0]))
##    print('(yMax-intercepts2[piece][pos]): ',(yMax-intercepts2[piece][pos]))
##    print("intercept: ",intercepts2[piece][pos])
##    print("grad: ",gradients2[piece][pos])

    
##    for i in range(int(disk_x[pos]),int(pieces_matrix[piece][0])):
##        #cv2.circle(img,(int(-i*gradients2[piece][pos]+(yMax-intercepts2[piece][pos])),int(i)), 2, (0,255,255), -1)#/gradients2[piece][pos
##        cv2.circle(img,(int(i),int(-i*gradients2[piece][pos]+yMax-intercepts2[piece][pos])), 2, (0,255,255), -1)
##    for i in range(int(b1.xMean),int(b1.xNorm)):
##        #cv2.circle(img,(int(-i*gradients2[piece][pos]+(yMax-intercepts2[piece][pos])),int(i)), 2, (0,255,255), -1)#/gradients2[piece][pos
##        cv2.circle(img,(int(i),100), 2, (0,255,255), -1)
    cv2.line(img,(int(pieces_matrix[piece][0]),int(pieces_matrix[piece][1])),(int(pieces_matrix[19+hole][0]),int(pieces_matrix[19+hole][1])),(0,255,0),1,1)
    cv2.line(img,(int(pieces_matrix[piece][0]),int(pieces_matrix[piece][1])),(int(disk_x[pos]),int(disk_y[pos])),(0,255,0),1,1)
    cv2.imshow('img: ',imutils.resize(img,width=700))
    cv2.imwrite('linesfinal'+str(frame)+str(piece)+'.png',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
