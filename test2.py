import cv2
import imutils
import numpy as np
import transform
import Image
from pytesseract import image_to_string
from scipy import spatial
from scipy.spatial import distance as dist
from operator import itemgetter
from matplotlib import pyplot as plt

#party moments
Wheel = cv2.imread('1.jpg',0)
Star = cv2.imread('2.jpg',0)
Home=cv2.imread('3.jpg',0)
Circle=cv2.imread('4.jpg',0)
Tree = cv2.imread('5.jpg',0)
Bicycle = cv2.imread('6.jpg',0)
Shell=cv2.imread('7.jpg',0)
Swan=cv2.imread('8.jpg',0)
Bell = cv2.imread('9.jpg',0)
Beatle=cv2.imread('10.jpg',0)
Elephant=cv2.imread('11.jpg',0)
w, h = Wheel.shape[::-1]

partyArray = [Wheel,Star,Home,Circle,Tree,Bicycle,Shell,Swan,Bell,Beatle,Elephant]

img_1 = cv2.imread('001.jpg')
img_2 = cv2.imread('002.jpg')
img_3 = cv2.imread('003.png')

img = img_1
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = np.ones((3,3),np.float32)/9
blur_image = cv2.filter2D(gray_image,-1,kernel)

binary_image_gray = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,5)
binary_image_blur = cv2.adaptiveThreshold(blur_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,5)

final = cv2.medianBlur(binary_image_blur,5)

num =1
font = cv2.FONT_HERSHEY_SIMPLEX
voted = False;


contours, hierarchy = cv2.findContours(binary_image_gray.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02* peri, True)
    area = cv2.contourArea(c)

    if ((len(approx) == 4) & (area>1000000) & (area<2000000)):
        cv2.drawContours(img, [c], -1, (0, 255, 0), 3)
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box)
        box = np.array(box, dtype="int")
        rect = transform.order_points(box)
        x1,y1 = rect[0]
        x2,y2 = rect[2]
        hig = x2-x1
        wid = y2-y1
        warped = transform.four_point_transform(binary_image_gray, box)
        warped2 = transform.four_point_transform(final, box)
        warped3 = transform.four_point_transform(img, box)
        arr=[]
        number=0
        contours2, hierarchy2 = cv2.findContours(warped2.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for c1 in contours2:
            peri1 = cv2.arcLength(c1, True)
            approx1 = cv2.approxPolyDP(c1, 0.02* peri1, True)
            area1 = cv2.contourArea(c1)
            #print(area1)    
            if ((len(approx1) == 4) & (area1>20000) & (area1<28000)):
                                
                box1 = cv2.minAreaRect(c1)
                box1 = cv2.cv.BoxPoints(box1)
                box1 = np.array(box1, dtype="int")
                #cv2.drawContours(warped3, [box1], -1, (0, 255, 0), 3)                 
                rect1 = transform.order_points(box1)
                xa1,ya1 = rect1[0]
                xa2,ya2 = rect1[2]
                #subimage = warped3[(int)(ya1):(int)(ya2),(int)(xa1):(int)(xa2)]

                cv2.rectangle(warped3,(xa2,ya1),((int)(xa2+120),ya2),(255,0,0),3)

                subimage1 = warped2[(int)(ya1+10):(int)(ya2-10),(int)(xa2+10):(int)(xa2+110)]

                number=(int)((ya1)/h)
                width, height = subimage1.shape[:2]
                area = width*height
                dst = cv2.inRange(subimage1,0, 50)
                no_black = cv2.countNonZero(dst)
                #print((float)(no_black)/area*100)
                if((float)(no_black)/area*100>5):
                    voted = True
                    cv2.putText(warped3,'Voted',((int)(xa2+20),(int)(ya1+20)), font, 1,(0,0,255),3)
                    arr.append(number)
                
        for k in range(11):
            res = cv2.matchTemplate(warped.copy(),partyArray[k],cv2.TM_CCOEFF_NORMED)
            threshold = 0.55
            loc = np.where( res >= threshold)

            for pt in zip(*loc[::-1]):
                cv2.rectangle(warped3, pt, (pt[0] + w, pt[1] + h), (255,0,255), 1)
            
            numb = (int)(pt[1]/h)
            if(numb in arr):
                if(k==0):
                    print('Voted to Wheel')
                elif(k==1):
                    print('Voted to Star')
                elif(k==2):
                    print('Voted to Home')
                elif(k==3):
                    print('Voted to Circle')
                elif(k==4):
                    print('Voted to Tree')
                elif(k==5):
                    print('Voted to Bicycle')
                elif(k==6):
                    print('Voted to Shell')
                elif(k==7):
                    print('Voted to Swan')
                elif(k==8):
                    print('Voted to Bell')
                elif(k==9):
                    print('Voted to Beatle')
                elif(k==10):
                    print('Voted to Elephant') 

              

    if ((len(approx) == 4) & (area>200000) & (area<300000)):
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box)
        box = np.array(box, dtype="int")
        #cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
        rect = transform.order_points(box)

        war = transform.four_point_transform(img, box)
        finall = cv2.medianBlur(war,5)
        war2 = transform.four_point_transform_Special(binary_image_gray, box)
        
        conto,hierar = cv2.findContours(war2.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in conto:
            #cv2.drawContours(war, [cnt], -1, (0, 255, 0), 2)
            peri2 = cv2.arcLength(cnt, True)
            approx2 = cv2.approxPolyDP(cnt, 0.02* peri2, True)
            area2 = cv2.contourArea(cnt)
            if ((len(approx2) == 4) & (area2>200000) & (area2<300000)):
                #cv2.drawContours(war, [cnt], -1, (0, 255, 0), 2)
                box2 = cv2.minAreaRect(cnt)
                box2 = cv2.cv.BoxPoints(box2)
                box2 = np.array(box2, dtype="int")
                rect2 = transform.order_points(box2)
        
                x1,y1 = rect2[0]
                x2,y2 = rect2[2]
                unitLength = (int)(x2-x1)/10;
                unitWidth = (int)(y2-y1)/4;
                
                voteCount = 0
                for j in range(4):
                    for i in range(10):
                        if((j*10+i+1)==26):
                            continue
                        if((j*10+i+1)==27):
                            continue
                        subimagef = war[j*unitWidth+5:(j+1)*unitWidth-10, i*unitLength+12:(i+1)*unitLength-12]
                        #cv2.imshow(str(j*10 +i+1), subimagef)
                

                        cv2_im = cv2.cvtColor(subimagef,cv2.COLOR_BGR2RGB)
                        pil_im = Image.fromarray(cv2_im)
                        text = image_to_string(pil_im,config='-psm 7')
                        if(text != str(j*10 +i+1)):
                            voteCount+=1
                            xa = (int)(i*unitLength)
                            ya = (int)(j*unitWidth)
                            cv2.rectangle(finall,(xa,ya),(xa+unitLength-5,ya+unitWidth-5),(0,0,255),3)
                            print('Vote '+str(voteCount)+' goes to ' +str(j*10 +i+1))
                  
            break                   
        
        result = imutils.resize(finall, width=400)
        cv2.imshow('lower area', result)
        
result = imutils.resize(warped3, width=400)
cv2.imshow('Upper area', result)        
    
#cv2.imwrite('final.png',img)
resized = imutils.resize(img, width=400)
#cv2.imshow('ori2', resized)
cv2.waitKey(0)
