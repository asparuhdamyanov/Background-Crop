import cv2, numpy as np, glob, os

SCRIPT_DIR = os.path.join(os.path.dirname(__file__))

for imagePath in glob.glob(os.path.join(SCRIPT_DIR, '80p', '80p', '*.jp*g')):

    image = cv2.imread(imagePath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 20, 40)

    x,y,w,h = cv2.boundingRect(edges)
    padding = 10
    x -= padding
    w += padding * 2
    y -= padding
    h += padding * 2
    x = max(round(x), 0)
    w = round(w) if x + w < image.shape[1] else image.shape[1] - x
    y = max(round(y), 0)
    h = round(h) if y + h < image.shape[0] else image.shape[0] - y

    percentages = {}
    percentages['top'] = y/image.shape[0]
    percentages['bottom'] = (image.shape[0]-(y+h))/image.shape[0]
    percentages['left'] = x/image.shape[1]
    percentages['right'] = (image.shape[1]-(x+w))/image.shape[1]
    
    print('\n' + '='*5, 'RESULT', '='*5)
    for key, value in percentages.items():
        print(f'{key:>8}', f'{value:6.2%}')

    view = cv2.rectangle(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), (x,y), (x+w,y+h), color=(0,255,0), thickness=2)
    if view.shape[0] > 1000:
        view = cv2.resize(view, (int(view.shape[1]/2.5), int(view.shape[0]/2.5)))

    crop = image[y:y+h,x:x+w]
    if crop.shape[0] > 1000:
        crop = cv2.resize(crop, (int(crop.shape[1]/2.5), int(crop.shape[0]/2.5)))

    cv2.imshow('canny', view)
    cv2.imshow('crop', crop)  
    cv2.waitKey(0)
