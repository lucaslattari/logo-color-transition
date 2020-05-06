import numpy as np
import cv2, random, operator
from tqdm import tqdm

def showImage(img):
    from matplotlib import pyplot as plt
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

def hexToRGB(hexColor):
    h = hexColor.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def generateRGBArray():
    best_colors_hex = [
        '#143D59', '#F4B41A', '#213970', '#210070', '#FFE042', '#E71989', '#5B0E2D', '#FFA781',
        '#5E001F', '#00E1D9', '#F5A21C', '#030F4F', '#250C77', '#ED642B', '#FFE01B', '#000000',
        '#FF6E0C', '#F20C90', '#0046BF', '#FEEF22', '#F26764', '#FFFFFF', '#687818', '#FFD58E',
        '#072F54', '#FBC108', '#FF6600', '#000000', '#70C19A', '#939393', '#F224F2', '#FFFFFF',
        '#2D2D2B', '#EC9347', '#2F455C', '#1DCDFE', '#EE9142', '#265B94', '#91C11E', '#FFFFFF',
        '#00BCB0', '#5630FF'
        ]

    best_colors_rgb = []
    for c in best_colors_hex:
        best_colors_rgb.append(hexToRGB(c))

    return best_colors_rgb

def getRandomColor(best_colors_rgb):
    randomColor = random.randrange(len(best_colors_rgb))
    if randomColor % 2 == 0:
        randomColor2 = randomColor + 1
    else:
        randomColor2 = randomColor - 1

    frontColor = randomColor
    backColor = randomColor2

    return frontColor, backColor

def generateRandomLogo(width, height, frontColor, backColor, logoBW):
    logoImage = np.zeros((height, width, 3), np.uint8)

    for y in range(0, height):
        for x in range(0, width):
            if logoBW[y][x] == 0:
                logoImage[y][x][0] = frontColor[0]
                logoImage[y][x][1] = frontColor[1]
                logoImage[y][x][2] = frontColor[2]
            else:
                logoImage[y][x][0] = backColor[0]
                logoImage[y][x][1] = backColor[1]
                logoImage[y][x][2] = backColor[2]

    return logoImage

def getInterpColor(firstColor, secondColor, tInterp):
    colorTInterp = np.zeros((3))
    if firstColor[0] > secondColor[0]:
        colorTInterp[0] = firstColor[0] - tInterp[0]
    else:
        colorTInterp[0] = firstColor[0] + tInterp[0]
    #------------------------------------------------
    if firstColor[1] > secondColor[1]:
        colorTInterp[1] = firstColor[1] - tInterp[1]
    else:
        colorTInterp[1] = firstColor[1] + tInterp[1]
    #------------------------------------------------
    if firstColor[2] > secondColor[2]:
        colorTInterp[2] = firstColor[2] - tInterp[2]
    else:
        colorTInterp[2] = firstColor[2] + tInterp[2]

    return colorTInterp

def computeLogoAnimation(maxIndex, diffColorArray, firstFrontColor, secondFrontColor, firstBackColor, secondBackColor, logoBW, filenameVideo):
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter(filenameVideo, fourcc, 20.0, (wLogo, hLogo * 5))
    frames = 0

    maxValueDiffArray = diffColorArray[maxIndex]
    for i in tqdm(range(0, maxValueDiffArray)):
        startInterpFrontB = diffColorArray[0] / maxValueDiffArray
        startInterpFrontG = diffColorArray[1] / maxValueDiffArray
        startInterpFrontR = diffColorArray[2] / maxValueDiffArray
        startInterpBackB = diffColorArray[3] / maxValueDiffArray
        startInterpBackG = diffColorArray[4] / maxValueDiffArray
        startInterpBackR = diffColorArray[5] / maxValueDiffArray

        endInterpFrontB = 1.0 - startInterpFrontB
        endInterpFrontG = 1.0 - startInterpFrontG
        endInterpFrontR = 1.0 - startInterpFrontR
        endInterpBackB = 1.0 - startInterpBackB
        endInterpBackG = 1.0 - startInterpBackB
        endInterpBackR = 1.0 - startInterpBackB

        currentInterp = i / maxValueDiffArray

        tInterpFrontB = int(diffColorArray[0] * currentInterp)
        tInterpFrontG = int(diffColorArray[1] * currentInterp)
        tInterpFrontR = int(diffColorArray[2] * currentInterp)
        tInterpBackB = int(diffColorArray[3] * currentInterp)
        tInterpBackG = int(diffColorArray[4] * currentInterp)
        tInterpBackR = int(diffColorArray[5] * currentInterp)

        colorTInterFront = getInterpColor(firstFrontColor, secondFrontColor, [tInterpFrontB, tInterpFrontG, tInterpFrontR])
        colorTInterBack = getInterpColor(firstBackColor, secondBackColor, [tInterpBackB, tInterpBackG, tInterpBackR])

        #print(colorTInterFront, [tInterpFrontB, tInterpFrontG, tInterpFrontR])

        hNewImage, wNewImage = logoBW.shape
        newImage = np.zeros((hNewImage * 5, wNewImage, 3), np.uint8)
        newImage[:] = colorTInterBack

        for y in range(0, hNewImage):
            for x in range(0, wNewImage):
                if logoBW[y][x] == 0:
                    newImage[int(y + hNewImage * 3.5)][x] = colorTInterFront[0], colorTInterFront[1], colorTInterFront[2]
                else:
                    newImage[int(y + hNewImage * 3.5)][x] = colorTInterBack[0], colorTInterBack[1], colorTInterBack[2]

        out.write(newImage)
        #cv2.imwrite(str(frames) + ".png", newImage)
        #print(firstFrontColor, colorTInterFront, secondFrontColor)
        #print(firstBackColor, colorTInterBack, secondBackColor)

        frames += 1
    out.release()

    '''
        print("front b", firstFrontColor[0], secondFrontColor[0], tInterpFrontB, startInterpFrontB, endInterpFrontB)
        print("front g", firstFrontColor[1], secondFrontColor[1], tInterpFrontG, startInterpFrontG, endInterpFrontG)
        print("front r", firstFrontColor[2], secondFrontColor[2], tInterpFrontR, startInterpFrontR, endInterpFrontR)
        print("back b", firstBackColor[0], secondBackColor[0], tInterpBackB, startInterpBackB, endInterpBackB)
        print("back g", firstBackColor[1], secondBackColor[1], tInterpBackG, startInterpBackG, endInterpBackG)
        print("back r", firstBackColor[2], secondBackColor[2], tInterpBackR, startInterpBackR, endInterpBackR)
        print("current interp constant", i / maxValueDiffArray)
    '''

#carrega logo
logo = cv2.imread("ud.png", 0)
hLogo, wLogo = logo.shape

dim = (int(wLogo * 0.5), int(hLogo * 0.5))
logo = cv2.resize(logo, dim)
hLogo, wLogo = logo.shape

_, logoBW = cv2.threshold(logo, 127, 255, cv2.THRESH_BINARY)

best_colors_rgb = generateRGBArray()

totalVideos = 0
while(1):
    #gera primeiro logo
    firstFrontColorIndex, firstBackColorIndex = getRandomColor(best_colors_rgb)
    if totalVideos == 0:
        firstFrontColor, firstBackColor = best_colors_rgb[firstFrontColorIndex], best_colors_rgb[firstBackColorIndex]
    else:
        firstFrontColor = secondFrontColor
        firstBackColor = secondBackColor
    firstLogoImage = generateRandomLogo(wLogo, hLogo, firstFrontColor, firstBackColor, logoBW)

    #gera segundo logo
    secondFrontColorIndex, secondBackColorIndex = getRandomColor(best_colors_rgb)
    secondFrontColor, secondBackColor = best_colors_rgb[secondFrontColorIndex], best_colors_rgb[secondBackColorIndex]
    while firstFrontColorIndex == secondFrontColorIndex or firstBackColorIndex == secondBackColorIndex:
        secondFrontColorIndex, secondBackColorIndex = getRandomColor(best_colors_rgb)
        secondFrontColor, secondBackColor = best_colors_rgb[secondFrontColorIndex], best_colors_rgb[secondBackColorIndex]
    secondLogoImage = generateRandomLogo(wLogo, hLogo, secondFrontColor, secondBackColor, logoBW)

    #computa a diferença entre as cores
    diffFront = map(lambda x, y: abs(x - y), firstFrontColor, secondFrontColor)
    diffFrontB, diffFrontG, diffFrontR = diffFront
    diffBack = map(lambda x, y: abs(x - y), firstBackColor, secondBackColor)
    diffBackB, diffBackG, diffBackR = diffBack
    diffArray = np.array([diffFrontB, diffFrontG, diffFrontR, diffBackB, diffBackG, diffBackR])
    maxIndices = np.where(diffArray == np.amax(diffArray))
    #print(len(maxIndices), diffArray)

    if(len(maxIndices) == 1):
        maxIndex = maxIndices[0]
    else:
        maxIndex = maxIndices[0][0]
    print(maxIndex, type(maxIndex))

    computeLogoAnimation(maxIndex[0], diffArray, firstFrontColor, secondFrontColor, firstBackColor, secondBackColor, logoBW, str(totalVideos) + ".avi")
    totalVideos += 1
