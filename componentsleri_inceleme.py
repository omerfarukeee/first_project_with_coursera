import numpy as np
import cv2

prior = 2
num_levels = 4
num_histogram_bins = 5

camera =cv2.VideoCapture('C:/Users/ASUS/Desktop/EGITIM_Uni/Uni_DERSLER/Bitirme_Projesi/Yol_Videoları/IMG_8407.MOV')
#camera =cv2.VideoCapture('C:/Users/ASUS/Desktop/Bitirme Projesi/Open CV Resim ve Videolar/serit1.mp4')
kernel = np.ones((2,2),np.uint8)

while True:
    # Loading Camera
    ret, frame = camera.read()
    frame = cv2.resize(frame, (480, 320))

    blurred = cv2.pyrMeanShiftFiltering(frame, 3, 3)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    converted_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    img_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height,width,channels = converted_img.shape
    num_superpixels_new = 1000       #cv2.getTrackbarPos('Number of Superpixels', 'SEEDS')
    num_iterations = 5              #cv2.getTrackbarPos('Iterations', 'SEEDS')
    result=label=mask=img_gray      #if not seeds or num_superpixels_new != num_superpixels:
    num_superpixels = num_superpixels_new
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels,
                num_superpixels, num_levels, prior, num_histogram_bins)
    color_img = np.zeros((height,width,3), np.uint8)
    color_img[:] = (255, 0, 0)

    seeds.iterate(converted_img, num_iterations)

    # retrieve the segmentation result
    labels = seeds.getLabels()

    # labels output: use the last x bits to determine the color
    num_label_bits = 2
    labels &= (1<<num_label_bits)-1
    labels *= 1<<(16-num_label_bits)

    mask = seeds.getLabelContourMask(False)
    """edge = cv2.Canny(mask, 255, 255)"""
    cv2.imshow("input",frame)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(edge, contours, -1, (255, 255, 0), 1)
    cv2.imshow("mask",mask)
    final=frame
    for i in range(0,len(contours)):
        mask2=np.zeros((320,480),np.uint8)
        cv2.drawContours(mask2,contours,i,(255,255,255),cv2.FILLED)
        meancolor=cv2.mean(converted_img,mask2)
        print(meancolor)

        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        # find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask2, connectivity=8)
        # connectedComponentswithStats yields every seperated component with information on each of them, such as size
        # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1];
        nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 10
        max_size = 10000

        # your answer image
        mask2 = np.zeros((output.shape))
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if max_size >= sizes[i] >= min_size:
                mask2[output == i + 1] = 255
                

        cv2.imshow("mask2",mask2)
        cv2.waitKey()

    """x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow("bolge", final)
    #if display_mode == 0:
    cv2.imshow('SEEDS0', result)
    #elif display_mode == 1:
    cv2.imshow('SEEDS1', mask)
    #else:
    cv2.imshow('SEEDS2', labels)"""

    cv2.waitKey()
    if cv2.waitKey(25) & 0XFF == ord('q'):
         break

camera.release()
cv2.destroyAllWindows()
