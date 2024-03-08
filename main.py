# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def readdata():
    data = []
    with open("C:\\Users\\omara\\OneDrive - University of Vermont\\CrowdHuman_Dataset\\annotation_train.odgt") as file:
        for line in file:
            data.append(json.loads(line))
    return data


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = readdata()

    #print the annotation based on image
    #print(df['ID' == "273271,1bab200041e8121f"])
    ID = "273271,1bab200041e8121f"

    image_path = (f"C:\\Users\\omara\\OneDrive - University of Vermont\\CrowdHuman_Dataset\\CrowdHuman_train01\\"
                  f"Images\\{ID}.JPG")
    img = cv2.imread(image_path)
    cv2.imshow(f"{ID}", img)
    cv2.waitKey(0)
