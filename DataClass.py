from Libraries import *


def readdata(initpath, filename) -> list:
    data = []
    filepath = os.path.join(initpath, filename)
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))  # Parse each line separately and append to list
    return data


def readimages(data, initpath) -> list:
    images = []
    for i in range(min(5, len(data))):  # Process a maximum of 5 images for demonstration
        ID = data[i]['ID']
        img = None
        for folder in ["Images", "Images 2", "Images 3"]:
            img_path = os.path.join(initpath, folder, f"{ID}.JPG")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                break
        if img is not None:
            images.append(img)
        else:
            print(f"Image {ID} not found in any directory.")
    return images


