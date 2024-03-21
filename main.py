# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2


def print_hi(name) -> None:
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def readdata(initpath) -> list:
    data = []
    with open(f"{initpath}annotation_train.odgt") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def displayimages(df, count, initpath) -> None:

    images = []
    for i in range(count):
        ID = df[i]['ID']
        print(df[i])
        paths = (f"{initpath}CrowdHuman_train01\\Images\\{ID}.JPG",
                 f"{initpath}CrowdHuman_train02\\Images\\{ID}.JPG",
                 f"{initpath}CrowdHuman_train03\\Images\\{ID}.JPG")
        for path in paths:
            img = cv2.imread(path)
            if(img is not None):
                cv2.imshow(ID, img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    initpath = "C:\\Users\\omara\\OneDrive - University of Vermont\\CrowdHuman_Dataset\\"
    df = readdata(initpath)
    displayimages(df, 5, initpath)
