import cv2
import numpy as np
import utils

####
path = "1.jpg"
widthImg = 700
heightImg = 700
questions = 5
choices = 5
Correctans = [1,2,0,0,4]

# LOAD IMAGE 
img = cv2.imread(path)
h, w = img.shape[:2]

# Détecter orientation et ajuster
if w > h: 
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

img = cv2.resize(img, (widthImg, heightImg))
imgCountours = img.copy()
imgFinal = img.copy()
imgBiggestCountours = img.copy()

# PREPROCESSING 
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 70)

#  FIND CONTOURS 
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgCountours, contours, -1, (0,255,0), 10)

# Rectangles
rectCon = utils.rectCountour(contours)
biggestContour = utils.getCornerPoints(rectCon[0])
gradePoints = utils.getCornerPoints(rectCon[1])

if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestCountours, biggestContour, -1, (0,255,0), 20)
    cv2.drawContours(imgBiggestCountours, gradePoints, -1, (255,0,0), 20)

    biggestContour = utils.reorder(biggestContour)
    gradePoints = utils.reorder(gradePoints)

    # Perspective transform pour l'examen
    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    # Perspective transform pour le score
    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0,0],[325,0],[0,150],[325,150]])
    matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
    imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325,150))

    # Threshold
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray, 150, 255, cv2.THRESH_BINARY_INV)[1]

    # Split boxes
    boxes = utils.splitBoxes(imgThresh)

    # Analyse des pixels
    myPixelValues = np.zeros((questions, choices))
    countC = 0
    countR = 0

    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelValues[countR][countC] = totalPixels
        countC += 1
        if countC == choices:
            countR += 1
            countC = 0

    # Détection des réponses choisies
    myIndex = []
    for x in range(questions):
        arr = myPixelValues[x]
        myIndexVal = np.where(arr == np.amax(arr))
        myIndex.append(int(myIndexVal[0][0]))

    # Calcul du score
    grading = [1 if Correctans[x] == myIndex[x] else 0 for x in range(questions)]
    score = (sum(grading)/questions)*100
    print("Score:", score)

    # Affichage des réponses
    imgResult = imgWarpColored.copy()
    imgResult = utils.showAnswers(imgResult, myIndex, grading, Correctans, questions, choices)

    imRawDrawing = np.zeros_like(imgWarpColored)
    imRawDrawing = utils.showAnswers(imRawDrawing, myIndex, grading, Correctans, questions, choices)
    invmatrix = cv2.getPerspectiveTransform(pt2, pt1)
    imgInvWrap = cv2.warpPerspective(imRawDrawing, invmatrix, (widthImg, heightImg))

    imgRawGrade = np.zeros_like(imgGradeDisplay)
    cv2.putText(imgRawGrade, str(int(score))+"%", (60,100), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0), 3)
    invmatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
    imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invmatrixG, (widthImg, heightImg))

    # Fusion des images
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWrap, 1, 0)
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

#  STACK IMAGES
imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur, imgCanny],
              [imgCountours, imgBiggestCountours, imgWarpColored, imgThresh],
              [imgResult, imRawDrawing, imgInvWrap, imgFinal])
labels = [["Original","Gray","Blur","Canny"],
          ["Contours","Biggest Contour","Warped","Threshold"],
          ["Result","Raw Drawing","Inverse Wrap","Final"]]

stackedImages = utils.stackImages(imageArray, 0.3, labels)

cv2.imshow("Stacked Images", stackedImages)
cv2.imshow("Final Result", imgFinal)
cv2.waitKey(0)
