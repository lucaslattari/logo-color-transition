import numpy as np
import cv2, random, glob, sys, os.path
from tqdm import tqdm
from moviepy.editor import VideoFileClip, concatenate_videoclips
from argparse import ArgumentParser

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

def computeSingleLogoAnimation(maxIndex, diffColorArray, firstFrontColor, secondFrontColor, firstBackColor, secondBackColor, logoBW, filenameVideo, args):
    hLogo, wLogo = logoBW.shape
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter(filenameVideo, fourcc, 20.0, (int(wLogo * args.widthLogo), int(hLogo * args.heightLogo)))
    frames = 0

    maxValueDiffArray = diffColorArray[maxIndex]
    print(diffColorArray, maxValueDiffArray)
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

        newImage = np.zeros((int(hLogo * args.heightLogo), int(wLogo * args.widthLogo), 3), np.uint8)
        newImage[:] = colorTInterBack

        for y in range(0, hLogo):
            for x in range(0, wLogo):
                if logoBW[y][x] == 0:
                    newImage[int(y + hLogo * args.positionHeightLogo)][int(x + wLogo * args.positionWidthLogo)][0] = colorTInterFront[0]
                    newImage[int(y + hLogo * args.positionHeightLogo)][int(x + wLogo * args.positionWidthLogo)][1] = colorTInterFront[1]
                    newImage[int(y + hLogo * args.positionHeightLogo)][int(x + wLogo * args.positionWidthLogo)][2] = colorTInterFront[2]

        out.write(newImage)

        frames += 1

    #para a transição não ficar tão abrupta
    for i in range(0, 10 * 20):
        newImage = np.zeros((int(hLogo * args.heightLogo), int(wLogo * args.widthLogo), 3), np.uint8)
        newImage[:] = secondBackColor

        for y in range(0, hLogo):
            for x in range(0, wLogo):
                if logoBW[y][x] == 0:
                    newImage[int(y + hLogo * args.positionHeightLogo)][int(x + wLogo * args.positionWidthLogo)] = secondFrontColor
        out.write(newImage)
    out.release()

def getBinaryLogo(logoFilename):
    #carrega logo
    logo = cv2.imread(logoFilename, 0)
    hLogo, wLogo = logo.shape

    dim = (int(wLogo * 0.5), int(hLogo * 0.5))
    logo = cv2.resize(logo, dim)

    _, logoBW = cv2.threshold(logo, 127, 255, cv2.THRESH_BINARY)

    return logoBW

def mergeClips():
    clipFilenames = glob.glob("*.avi")
    clipFiles = []

    for i in tqdm(range(0, len(clipFilenames))):
        clipFiles.append(VideoFileClip(str(i) + ".avi"))

    final_clip = concatenate_videoclips(clipFiles)
    final_clip.write_videofile("final.mp4")

def computeEntireAnimation(args):
    logoBW = getBinaryLogo(args.logoFilename)
    hLogo, wLogo = logoBW.shape

    best_colors_rgb = generateRGBArray()

    for countVideos in range(0, args.countTransitions):
        #gera primeiro logo
        firstFrontColorIndex, firstBackColorIndex = getRandomColor(best_colors_rgb)
        if countVideos == 0:
            firstFrontColor, firstBackColor = best_colors_rgb[firstFrontColorIndex], best_colors_rgb[firstBackColorIndex]
        else:
            firstFrontColor = secondFrontColor
            firstBackColor = secondBackColor
        firstLogoImage = generateRandomLogo(wLogo, hLogo, firstFrontColor, firstBackColor, logoBW)

        #gera segundo logo
        secondFrontColorIndex, secondBackColorIndex = getRandomColor(best_colors_rgb)
        secondFrontColor, secondBackColor = best_colors_rgb[secondFrontColorIndex], best_colors_rgb[secondBackColorIndex]
        while firstFrontColor == secondFrontColor or firstBackColor == secondBackColor:
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

        if(len(maxIndices) == 1):
            maxIndex = maxIndices[0]
        else:
            maxIndex = maxIndices[0][0]
        print(maxIndex)

        computeSingleLogoAnimation(maxIndex[0], diffArray, firstFrontColor, secondFrontColor, firstBackColor, secondBackColor, logoBW, str(countVideos) + ".avi", args)

def parse_args():
    parser = ArgumentParser(description = 'Cria animação de transição de cores com logos')
    parser.add_argument('logoFilename', help = 'Caminho do logo (deve ser imagem binária com só 2 cores, preto e branco)')
    parser.add_argument('countTransitions', type = int, help = 'Total de transições que serão criadas')
    parser.add_argument('--h', action = 'store', dest = 'heightLogo', type = float, default = 5.0, required = False,
                        help = 'Altura do logo no vídeo')
    parser.add_argument('--w', action = 'store', dest = 'widthLogo', type = float, default = 1.0, required = False,
                        help = 'Largura do logo no vídeo')
    parser.add_argument('-ph', action = 'store', dest = 'positionHeightLogo', type = float, default = 3.0, required = False,
                        help = 'Altura do logo no vídeo')
    parser.add_argument('-pw', action = 'store', dest = 'positionWidthLogo', type = float, default = 0.0, required = False,
                        help = 'Largura do logo no vídeo')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()

def main():
    arguments = parse_args()

    if not os.path.exists(arguments.logoFilename):
        print(f'{arguments.logoFilename} não existe (not found)')
        return

    computeEntireAnimation(arguments)
    mergeClips()

if __name__ == "__main__":
    main()
