import cv2


def alertObject(objx1, objy1, objx2, objy2):
    print("Kijelolt pontok:")
    print("(" + str(objx1) + "," + str(objy1) + ")")
    print("(" + str(objx2) + "," + str(objy2) + ")")
    print("Nyomjon meg egy billentyut a folytatashoz!")

def alertROI(objx1, objy1, objx2, objy2, objx3, objy3, objx4, objy4):
    print("Kijelolt pontok:")
    print("(" + str(objx1) + "," + str(objy1) + ")")
    print("(" + str(objx2) + "," + str(objy2) + ")")
    print("(" + str(objx3) + "," + str(objy3) + ")")
    print("(" + str(objx4) + "," + str(objy4) + ")")
    print("Nyomjon meg egy billentyut a folytatashoz!")

def printObjectPoints(img, objx1, objy1, objx2, objy2):

    settingImg = cv2.circle(img.copy(), (objx1, objy1), 8, (0,0,255), -1)
    settingImg = cv2.circle(settingImg, (objx2, objy2), 8, (0,0,255), -1)
    settingImg = cv2.line(settingImg, (objx1, objy1), (objx2, objy2), (0,0,255), 2)

    return settingImg

def printROIPoints(img, objx1, objy1, objx2, objy2, objx3, objy3, objx4, objy4):
    settingImg = cv2.circle(img.copy(), (objx1, objy1), 8, (0,0,255), -1)

    settingImg = cv2.circle(settingImg, (objx2, objy2), 8, (0,0,255), -1)

    settingImg = cv2.circle(settingImg, (objx3, objy3), 8, (0,0,255), -1)

    settingImg = cv2.circle(settingImg, (objx4, objy4), 8, (0,0,255), -1)

    settingImg = cv2.line(settingImg, (objx1, objy1), (objx2, objy2), (0,0,255), 2)
    settingImg = cv2.line(settingImg, (objx2, objy2), (objx4, objy4), (0,0,255), 2)
    settingImg = cv2.line(settingImg, (objx4, objy4), (objx3, objy3), (0,0,255), 2)
    settingImg = cv2.line(settingImg, (objx3, objy3), (objx1, objy1), (0,0,255), 2)

    return settingImg

def subsort(part1, part2):
    if part1[0] > part1[1]:
        part1[0], part1[1] = part1[1], part1[0]
    if part2[0] > part2[1]:
        part2[0], part2[1] = part2[1], part2[0]
    return [part1[0], part1[1], part2[0], part2[1]]

def sortGivenPoints(one, two, three, four):
    unsorted = [one, two, three, four]
    print(unsorted)
    
    check = True
    while check:
        count = 0
        for x in range(0,len(unsorted)-1):
            if unsorted[x][1] > unsorted[x+1][1]:
                unsorted[x], unsorted[x+1] = unsorted[x+1], unsorted[x]
                count += 1
        if count == 0:
            check = False
    #print(unsorted)
    return subsort([unsorted[0], unsorted[1]], [unsorted[2], unsorted[3]])


