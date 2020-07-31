import numpy as np
import cv2
from human_detection import is_human
import math
import time

def getForegroundMask(frame, background, th):
    # reduce the nois in the farme
    frame =  cv2.blur(frame, (5,5))
    # get the absolute difference between the foreground and the background
    fgmask= cv2.absdiff(frame, background)
    # convert foreground mask to gray
    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
    # apply threshold (th) on the foreground mask
    _, fgmask = cv2.threshold(fgmask, th, 255, cv2.THRESH_BINARY)
    # setting up a kernal for morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # apply morpholoygy on the foreground mask to get a better result
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    return fgmask

def MOG2init(history, T, nMixtures):
    # create an instance of MoG and setting up its history length
    fgbg = cv2.createBackgroundSubtractorMOG2(history)
    # setting up the protion of the background model
    fgbg.setBackgroundRatio(T)
    # setting up the number of MoG
    fgbg.setNMixtures(nMixtures)
    return fgbg

def extract_objs2(im, min_w=25, min_h=25, max_w=500, max_h=500, draw = None):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    arr = cv2.dilate(im, kernel, iterations=2)
    arr = np.array(arr, dtype=np.uint8)
    # cv2.imshow("arr",arr) 
    _, th = cv2.threshold(arr,127,255,0)
    # cv2.imshow("th",th) 
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    objs = []
    contours_list = []
    cv2.imwrite("tmp2.jpg", arr)
    for c in contours:
        contours_list.append(c)
        x,y,w,h = cv2.boundingRect(c)
        if (w >= min_w) & (w < max_w) & (h >= min_h ) & (h < max_h):
            objs.append([x,y,w,h, 1]) # The last one means that it is still needed to check
        else:
            print(w,h)
    return objs, contours_list

# this function returns static object map without pre-founded objects
def clean_map(m, o):
    rslt = np.copy(m)
    for i in range (0, len(o)):
        x, y= o[i][0], o[i][1]
        w, h= o[i][2], o[i][3]
        rslt[y:y+h, x:x+w] = 0
    return rslt

# print(is_human(cv2.imread("test_img.jpg")))

#cap = cv2.VideoCapture('AVSS_AB_EVAL_divx.avi')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


cap = cv2.VideoCapture(0)
time.sleep(2)
# cap = cv2.VideoCapture("rtsp://admin:HuaWei123@192.168.1.36:554/LiveMedia/ch1/Media2")

# background model
_, BG = cap.read()
# cv2.imwrite('bg_first_frame.jpg',BG)
# BG = cv2.imread('bg_first_frame.jpg')
# BG = cv2.flip(BG, -1)

# setting up a kernal for morphology
kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# MoG for long background model
fgbgl = MOG2init(300, 0.4, 3)
# MoG for short background model
fgbgs = MOG2init(300, 0.4, 3)

longBackgroundInterval = 10
shortBackgroundINterval = 1

clfg = longBackgroundInterval   # counter for longbackgroundInterval
csfg = shortBackgroundINterval  # counter for shortBackgroundInteral

# static obj likelihood
L = np.zeros(np.shape(cap.read()[1])[0:2])

static_obj_map = np.zeros(np.shape(cap.read()[1])[0:2])

# static obj likelihood constants
k, maxe, thh= 5, 2000, 800

# obj-extraction constants
slidewindowtime = 0
minwindowsize = 70
stepsize = 20
static_objs = []
static_contours = []
th_sp = 300 # a th for number of static pixels

# mask = np.zeros(BG.shape[:2], dtype = "uint8") # (576, 720, 3), take (576,720) same height and width as the image

# WHITE: Update pts which is the Area of Interest
pts = np.array([[280,0],[280,80],[150,160],[0,280],[0,550],[650,550],[650,0]], np.int32)
# pts = np.array([])
# cv2.fillPoly(mask,[pts],255,1)
while(1):
    _, frame = cap.read()
    # frame = cv2.flip(frame, -1)
    f2 = frame.copy()
    # frame = cv2.bitwise_and(frame,frame,mask=mask)
    if clfg == longBackgroundInterval:
        frameL = np.copy(frame)
        fgbgl.apply(frameL)
        BL = fgbgl.getBackgroundImage(frameL)
        clfg = 0
    else:
        clfg += 1

    if csfg == shortBackgroundINterval:
        frameS = np.copy(frame)
        fgbgs.apply(frameS)
        BS = fgbgs.getBackgroundImage(frameS)
        csfg = 0
    else:
        csfg += 1

    # update short&long foregrounds
    FL = getForegroundMask(frame, BL, 70)
    FS = getForegroundMask(frame, BS, 70)
    FG = getForegroundMask(frame, BG, 70)

    # detec static pixels and apply morphology on it
    static = FL&cv2.bitwise_not(FS)&FG
    static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, kernal)
    # cv2.imshow("static",static)
    # dectec non static objectes and apply morphology on it
    not_static = FS|cv2.bitwise_not(FL)
    not_static = cv2.morphologyEx(not_static, cv2.MORPH_CLOSE, kernal)
    # cv2.imshow("not_static",not_static)
    # update static obj likelihood
    L = (static == 255) * (L+1) + ((static == 255)^1) * L
    L = (not_static == 255) * (L-k) + ((not_static == 255)^1) * L
    L[ L>maxe ] = maxe
    L[ L<0 ] = 0

    # update static obj map
    static_obj_map = L
    # static_obj_map[L >= thh ] = 254
    # static_obj_map[L < thh ] = 0

    
    # if number of nonzero elements in static obj map greater than min window size squared there
    # could be a potential static obj, we will need to wait 200 frame to be pased if the condtion
    # still true we will call "extract_objs" function and try to find these objects.
    # cv2.imshow("clean_map(static_obj_map, static_objs)",clean_map(static_obj_map, static_objs))
    print("Pixel diff", np.count_nonzero(clean_map(static_obj_map, static_objs)))
    cv2.imshow("clean_map", clean_map(static_obj_map, static_objs))
    if(np.count_nonzero(clean_map(static_obj_map, static_objs)) > th_sp ):
        print("slidewindowtime",str(slidewindowtime))
        if(slidewindowtime > 200):
            new_objs, contours_list = extract_objs2(clean_map(static_obj_map, static_objs), draw = f2)
            # if we get new object, first we make sure that they are not dublicated ones and then
            # put the unique static objects in "static_objs" variable
            if(new_objs):
                for i in range(0, len(new_objs)):
                    if new_objs[i] not in static_objs:
                        static_objs.append(new_objs[i])
                        # static_contours.append(contours_list[i])
            if(contours_list):
                static_contours.append(contours_list)

            slidewindowtime = 0
        else:
            slidewindowtime += 1
    else:
            slidewindowtime = 0 if slidewindowtime < 0 else slidewindowtime - 1
    # draw recatngle around static obj/s

    c=0
    print("len(static_objs) : ",len(static_objs))
    for i in range (0, len(static_objs)):
        if(static_objs[i-c]):
            x, y = static_objs[i-c][0], static_objs[i-c][1]
            w, h = static_objs[i-c][2], static_objs[i-c][3]
            cen_x = (x + (x+w)) * 0.5
            cen_y = (y + (y+h)) * 0.5
            check_human_flag = static_objs[i-c][4]
            # check if the current static obj still in the scene 
            cv2.imshow("t", frame[y:y+h, x:x+w])
            # cv2.moveWindow('t',3000,-300)
            cv2.rectangle(f2, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.circle(f2, (int(cen_x), int(cen_y)), 7, (0,0,255), -1)
            cv2.imwrite("test_img.jpg", frame[y:y+h, x:x+w])
            if((np.count_nonzero(static_obj_map[y:y+h, x:x+w]) < w * h * .03)):
                static_objs.remove(static_objs[i-c])
                c += 1
                continue
            if(check_human_flag):
                if(check_human_flag > 25): # check if the founded obj is a human ever 1 sec
                    static_objs[i-c][4] = 0
                    if(is_human(frame[y-10:y+h+10, x-10:x+w+10])):
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                        print("is human")
                        continue
                    else:
                        print("no human")
                        continue
                else:
                    static_objs[i-c][4] += 1
             
    
    # boxes, weights = hog.detectMultiScale(frame, winStride=(8,8),scale=1.05 )
    # human = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    # for (xA, yA, xB, yB) in human:
    #         # display the detected boxes in the colour picture
    #     cen_xA = (xA + xB) * 0.5
    #     cen_yA = (yA + yB) * 0.5

    #     cv2.rectangle(f2, (xA, yA), (xB, yB), (0, 255, 0), 2)
    #     cv2.circle(f2, (int(cen_xA), int(cen_yA)), 7,  (0, 255, 0), -1)
    #     if len(static_objs) > 0:
    #         for obj in static_objs:
    #             x,y,w,h,_= obj
    #             cen_x = (x + (x+w)) * 0.5
    #             cen_y = (y + (y+h)) * 0.5

    #             distance = math.sqrt(abs(pow((cen_xA-cen_x),2)+pow((cen_yA-cen_y),2)))
    #             print(distance)
    #             cv2.line(f2,(int(cen_x),int(cen_y)),(int(cen_xA),int(cen_yA)),(255,0,0),2)
    #             cv2.putText(f2,str(int(distance)), (int((cen_x+cen_xA)/2),int((cen_y+cen_yA)/2)) , cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0, 0, 255),2,cv2.LINE_AA)


    # cv2.polylines(f2,[pts],True,(0,255,255))
    cv2.imshow("frame", f2)
    # cv2.moveWindow('frame',1450,-300)
    cv2.imshow("BG", BS)
    # cv2.moveWindow('BG',2200,-300)
    
    
    key = cv2.waitKey(1) & 0xff
    if key == 27:
        cv2.imwrite("f1.jpg", frame)
        cv2.imwrite("BG.jpg", BS)
        break


cap.release()
cv2.destoryAllWindows()
