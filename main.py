# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2


def print_hi(name)->None:
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def readdata(initpath) -> pd.DataFrame:
    data = []
    with open(f"{initpath}CrowdHuman_Dataset\\annotation_train.odgt") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def displayimages(df, count, initpath) -> None:
    Hori = []
    for i in range(count):
        ID = df[i]['ID']
        print(ID)
        '''
        try:
            image_path = (f"{initpath}CrowdHuman_Dataset\\"
                          f"CrowdHuman_train01\\Images\\{ID}.JPG")
            img = cv2.imread(image_path)
            try:
                image_path = (f"{initpath}CrowdHuman_Dataset\\"
                              f"CrowdHuman_train02\\Images\\{ID}.JPG")
                img = cv2.imread(image_path)
                try:
                    image_path = (f"{initpath}CrowdHuman_Dataset\\"
                                  f"CrowdHuman_train03\\Images\\{ID}.JPG")
                    img = cv2.imread(image_path)
                except:
                    print("file not found in any training directoy")
            except:
                print("file not found in 02 training directoy")
        except:
            print("file not found in 01 training directoy")

        Hori = np.insert(img)

    cv2.imshow(f"{ID}",Hori[:])
    cv2.waitKey(0)
'''

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    initpath = "C:\\Users\\omara\\OneDrive - University of Vermont\\"
    df = readdata(initpath)
    displayimages(df, 5, initpath)
