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
Wheel = [21091200.0, 0.0, 140762700.0, 0.038518518518518514, 9854274300.0, 0.0, 98718750.0, 0.18028846153846154, 67827702150.0, 0.0, 0.0, 0.0, 0.0, 0.0, 86821800.0, 6837222600.0, 23400.0, 1240200.0, 0.0, 0.0, 0.0, 2655900.0, 400163400.0, 21208660200.0]
Star  = [16631932.59828987, -859280.0742092133, 121193013.875, 0.03542951391577656, 7838004268.333333, 7407846.257575035, 91971443.94195402, 0.19591851600760873, 63725708052.25, 0.0001072062382751768, -98787.6168217063, 898333.8577764034, -0.0002104383975949327, 1.3000673915033716e-05, 68739164.25, 5001417178.7, 21666.5, 1062535.3333333333, -248776.50338745117, -3.6002897700729274e-06, -1.2435488153737702e-05, 2473300.1666666665, 374306556.4166666, 18341023281.733334]
Home = [24173874.666666657, 1.9073486328125e-06, 153124944.0, 0.04055059523809522, 11230082229.333334, 1.9073486328125e-06, 102091434.66666663, 0.17125382262996935, 69838769504.0, 2.0475937332234492e-17, 0.0, 1.9073486328125e-06, 0.0, 2.0475937332234492e-17, 99381258.66666666, 8198959944.0, 24416.0, 1355088.0, 1.52587890625e-05, 1.6380749865787594e-16, 2.0475937332234492e-17, 2759008.0, 413859338.6666666, 22969193296.0]
Circle = [20336022.293342426, -1437312.0481739044, 136871646.5833333, 0.03841565760442704, 9490719763.65, 10997940.900159836, 95698220.91685861, 0.18077822865577814, 65994344619.950005, 0.00013696645324356066, -157891.42355558276, 353243.5991706848, -0.00029826397603687464, 4.399234670255005e-06, 83906091.0, 6546865181.3, 23008.0, 1209388.3333333333, -2432705.4237060547, -3.0296492470381372e-05, -1.790003558209305e-05, 2606917.5, 391074561.8333333, 20531590461.516666]
Tree =[16154101.496516928, -119219.14059638977, 117998121.29166666, 0.03566796537871839, 7632198895.65, -2169945.7266407013, 88238806.0332377, 0.19482969568636024, 60312163259.450005, -3.2843035002410676e-05, -244799.60793705285, 7806072.397228479, -0.0005405131286630385, 0.0001181481664845239, 67841660.25, 4935506892.45, 21281.5, 1048803.5, -28341219.40448761, -0.00042895619438075957, -1.8044314931443806e-06, 2399292.833333333, 358736942.9166666, 17622051533.983334]
Bicycle = [11873045.333333328, 4.76837158203125e-07, 95780608.0, 0.03199404761904761, 5556000874.666667, 0.0, 80549205.33333331, 0.21705426356589141, 55102148416.0, 0.0, 0.0, 9.5367431640625e-07, 0.0, 1.8515434701327773e-17, 49168149.33333333, 3208226560.0, 19264.0, 847616.0, 7.62939453125e-06, 1.4812347761062218e-16, 9.257717350663886e-18, 2176832.0, 326531221.3333333, 14367373738.666666]
Shell = [9936940.75828144, -27047182.39799881, 74086208.45833333, 0.0499502090702615, 4463238877.083333, -100828260.0715437, 76675528.65768704, 0.38542633796346326, 48869511576.5, -0.004267642534720134, -385866.77048645914, -24210248.3377223, -0.0019396438328368015, -0.0010247195132484573, 38631201.916666664, 2611788079.5, 14104.5, 636174.6666666666, -683992475.1709518, -0.02895056781101676, -0.0011447951790909668, 1651105.3333333333, 269957730.9166666, 11985105740.75]
Swan = [15573450.0, 0.0, 115225200.0, 0.03481481481481482, 7298396175.0, 0.0, 89226562.5, 0.19946808510638298, 61305807712.5, 0.0, 0.0, 0.0, 0.0, 0.0, 64303050.0, 4581597600.0, 21150.0, 1015200.0, 0.0, 0.0, 0.0, 2400525.0, 361686150.0, 17360935200.0]
Bell = [24281793.75, 0.0, 154489106.25, 0.04037037037037037, 11330128987.5, 0.0, 103464843.75, 0.1720183486238532, 71088649368.75, 0.0, 0.0, 0.0, 0.0, 0.0, 99824925.0, 8235562443.75, 24525.0, 1361137.5, 0.0, 0.0, 0.0, 2783587.5, 419402025.0, 23276812387.5]
Beatle = [15201839.266069226, -41525621.17117119, 102705289.375, 0.05369187328860204, 7122636256.433333, -138585914.8734579, 90745610.04669303, 0.32050738801054146, 57932617205.8, -0.0037734154086740573, 452375.6094945371, -13422301.700920582, 0.0015977602103745463, -0.0003654622484858731, 60582022.08333333, 4683556591.25, 16826.5, 873836.1666666666, -884157820.3090134, -0.02407383712767674, -0.001130659049479415, 1968971.6666666665, 321147025.9166666, 16645137116.0]
Elephant = [20488631.25, 0.0, 138094031.25, 0.038148148148148146, 9575396287.5, 0.0, 97769531.25, 0.18203883495145629, 67175512706.25, 0.0, 0.0, 0.0, 0.0, 0.0, 84364725.0, 6580454343.75, 23175.0, 1216687.5, 0.0, 0.0, 0.0, 2630362.5, 396315675.0, 20806572937.5]


partyArray = [Wheel,Star,Home,Circle,Tree,Bicycle,Shell,Swan,Bell,Beatle,Elephant]

img_1 = cv2.imread('001.jpg')
img_2 = cv2.imread('002.jpg')
img_3 = cv2.imread('003.png')

img = img_3
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = np.ones((3,3),np.float32)/9
blur_image = cv2.filter2D(gray_image,-1,kernel)

binary_image_gray = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,5)
binary_image_blur = cv2.adaptiveThreshold(blur_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,5)

#ret,thresh1 = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)
#ret,thresh2 = cv2.threshold(blur_image,127,255,cv2.THRESH_BINARY)
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

        warped = transform.four_point_transform(binary_image_gray, box)
        warped2 = transform.four_point_transform(final, box)
        warped3 = transform.four_point_transform(img, box)
        
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
                cv2.drawContours(warped3, [box1], -1, (0, 255, 0), 3)                 
                rect1 = transform.order_points(box1)
                xa1,ya1 = rect1[0]
                xa2,ya2 = rect1[2]
                subimage = warped2[(int)(ya1+3):(int)(ya2-2),(int)(xa1+3):(int)(xa2-2)]
                            
                edged = cv2.Canny(subimage, 50, 100)
                edged = cv2.dilate(edged, None, iterations=1)
                #edged = cv2.erode(edged, None, iterations=1)

                cz,hierz = cv2.findContours(edged,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
                for ct in cz:
                    cv2.drawContours(edged,[ct],0,255,-1)
                edged = cv2.bitwise_not(edged)
                #cv2.imshow(str(num), edged)
                
                arr = []

                conto,hierar = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                for cnt in conto:
                    peri2 = cv2.arcLength(cnt, True)
                    approx2 = cv2.approxPolyDP(cnt, 0.02* peri2, True)
                    area2 = cv2.contourArea(cnt)
                    if(area2>11000):
                        M = cv2.moments(cnt)
                        for key, value in M.iteritems():
                            arr.append(value)   

                cv2.rectangle(warped3,(xa2,ya1),((int)(xa2+120),ya2),(255,0,0),3)

                num+=1
                #print(arr)
                distanceArr=[]
                for k in range(11):
                    result = dist.euclidean(arr,partyArray[k])
                    #print(result)
                    distanceArr.append(result)
                minindex = min(enumerate(distanceArr), key=itemgetter(1))[0]
                #print('------')
                #print(minindex)
                #print(distanceArr)
                #print('-------------------------------------------------------------')

                subimage1 = warped2[(int)(ya1+10):(int)(ya2-10),(int)(xa2+10):(int)(xa2+110)]
                #cv2.imshow('wraped'+str(num), subimage1)     
                width, height = subimage1.shape[:2]
                area = width*height
                dst = cv2.inRange(subimage1,0, 50)
                no_black = cv2.countNonZero(dst)
                #print((float)(no_black)/area*100)
                if((float)(no_black)/area*100>5):
                    voted = True
                    cv2.putText(warped3,'Voted',((int)(xa2+20),(int)(ya1+20)), font, 1,(0,0,255),3)
                if(minindex==0):
                    if(voted):
                        voted=False
                        print('Voted to Wheel')
                    cv2.putText(warped3,'Wheel',((int)(xa1+20),(int)(ya1+20)), font, 1,(0,0,255),3)
                elif(minindex==1):
                    if(voted):
                        voted=False
                        print('Voted to Star')
                    cv2.putText(warped3,'Star',((int)(xa1+20),(int)(ya1+20)), font, 1,(0,0,255),3)
                elif(minindex==2):
                    if(voted):
                        voted=False
                        print('Voted to Home')
                    cv2.putText(warped3,'Home',((int)(xa1+20),(int)(ya1+20)), font, 1,(0,0,255),3)    
                elif(minindex==3):
                    if(voted):
                        voted=False
                        print('Voted to Circle')
                    cv2.putText(warped3,'Circle',((int)(xa1+20),(int)(ya1+20)), font, 1,(0,0,255),3)    
                elif(minindex==4):
                    if(voted):
                        voted=False
                        print('Voted to Tree')
                    cv2.putText(warped3,'Tree',((int)(xa1+20),(int)(ya1+20)), font, 1,(0,0,255),3)    
                elif(minindex==5):
                    if(voted):
                        voted=False
                        print('Voted to Bicycle')
                    cv2.putText(warped3,'Bicycle',((int)(xa1+20),(int)(ya1+20)), font, 1,(0,0,255),3)    
                elif(minindex==6):
                    if(voted):
                        voted=False
                        print('Voted to Shell')
                    cv2.putText(warped3,'Shell',((int)(xa1+20),(int)(ya1+20)), font, 1,(0,0,255),3)    
                elif(minindex==7):
                    if(voted):
                        voted=False
                        print('Voted to Swan')
                    cv2.putText(warped3,'Swan',((int)(xa1+20),(int)(ya1+20)), font, 1,(0,0,255),3)    
                elif(minindex==8):
                    if(voted):
                        voted=False
                        print('Voted to Bell')
                    cv2.putText(warped3,'Bell',((int)(xa1+20),(int)(ya1+20)), font, 1,(0,0,255),3)    
                elif(minindex==9):
                    if(voted):
                        voted=False
                        print('Voted to Betel')
                    cv2.putText(warped3,'Betel',((int)(xa1+20),(int)(ya1+20)), font, 1,(0,0,255),3)    
                elif(minindex==10):
                    if(voted):
                        voted=False
                        print('Voted to Elephant')
                    cv2.putText(warped3,'Elephant',((int)(xa1+20),(int)(ya1+20)), font, 1,(0,0,255),3)

    if ((len(approx) == 4) & (area>200000) & (area<300000)):
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box)
        box = np.array(box, dtype="int")
        #cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
        rect = transform.order_points(box)

        war = transform.four_point_transform(img, box)
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
                        subimagef = war[j*unitWidth+5:(j+1)*unitWidth-12, i*unitLength+5:(i+1)*unitLength-12]
                        #cv2.imshow(str(j*10 +i+1), subimagef)
                
                        cv2_im = cv2.cvtColor(subimagef,cv2.COLOR_BGR2RGB)
                        pil_im = Image.fromarray(cv2_im)
                        text = image_to_string((pil_im),config='-psm 7')
                        if(text != str(j*10 +i+1)):
                            voteCount+=1
                            xa = (int)(i*unitLength)
                            ya = (int)(j*unitWidth)
                            cv2.rectangle(war,(xa,ya),(xa+unitLength,ya+unitWidth),(0,0,255),3)
                            print('Vote '+str(voteCount)+' goes to ' +str(j*10 +i+1))
                  
            break                   
        
        result = imutils.resize(war, width=400)
        cv2.imshow('lower area', result)
        
result = imutils.resize(warped3, width=400)
cv2.imshow('Upper area', result)        
    
#cv2.imwrite('final.png',img)
resized = imutils.resize(img, width=400)
#cv2.imshow('ori2', resized)
cv2.waitKey(0)
