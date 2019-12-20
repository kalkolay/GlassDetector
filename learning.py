from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import cv2
import numpy as np
import pickle
import random
import cvlib as cv
from cvlib.object_detection import draw_bbox
import httpx


def find_edges(img):
    img = cv2.bilateralFilter(img, 5, 75, 75)
    img = cv2.Canny(img, 75, 160)
    return img


imgs_GreenGlass = list()
imgs_BrownGlass = list()
imgs_WhiteGlass = list()


for i in range(1, 31):
    img = cv2.imread(f'GreenGlass/{i}.jpg', 0)
    img = find_edges(img)
    imgs_GreenGlass.append(img)

    img = cv2.imread(f'BrownGlass/{i}.jpg', 0)
    img = find_edges(img)
    imgs_BrownGlass.append(img)

    img = cv2.imread(f'WhiteGlass/{i}.jpg', 0)
    img = find_edges(img)
    imgs_WhiteGlass.append(img)


def preprocess(img, im_width=720, c_size=5):
    data = np.argwhere(img > 0)
    scaled_data = data

    model = PCA(n_components=2)
    model.fit(scaled_data)
    tdata = model.transform(scaled_data)

    im_width = 720

    points = (tdata - tdata.min(axis=0)) / ((tdata.max(axis=0) - tdata.min(axis=0)).max())

    offset = (1 - (points.max(axis=0) - points.min(axis=0))) / 2

    im_points = ((points + offset) * im_width).astype(int)

    image = np.zeros((im_width, im_width))
    for point in im_points:
        image = cv2.circle(image, tuple(point), c_size, 1, -1)
    return image


imgs_GreenGlass = [preprocess(img).flatten() for img in imgs_GreenGlass]
imgs_BrownGlass = [preprocess(img).flatten() for img in imgs_BrownGlass]
imgs_WhiteGlass = [preprocess(img).flatten() for img in imgs_WhiteGlass]

'''
print('Learning process started.')
while True:
    indices = list(range(30))
    random.shuffle(indices)

    all_data = np.vstack([
        np.array(imgs_WhiteGlass)[indices[:24]],
        np.array(imgs_BrownGlass)[indices[:24]],
        np.array(imgs_GreenGlass)[indices[:24]]
    ])

    labels = np.array([0] * 24 + [1] * 24 + [2] * 24)

    test = np.vstack([
        np.array(imgs_WhiteGlass)[indices[24:]],
        np.array(imgs_BrownGlass)[indices[24:]],
        np.array(imgs_GreenGlass)[indices[24:]]
    ])

    labels_test = np.array([0] * 6 + [1] * 6 + [2] * 6)  # predict size

    text_labels = np.array(['WhiteGlass', 'BrownGlass', 'GreenGlass'])

    reg = LogisticRegression(solver='lbfgs', multi_class='auto')
    reg.fit(all_data, labels)

    score = reg.score(test, labels_test)
    # print(f'score = {score}')
    if score > 0.6:
        print('Learning process finished.')
        
        # save the model to disk
        pickle.dump(reg, open('finalized_model.sav', 'wb'))

        break
'''

# load the model from disk
print('Loading model.')
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
indices = list(range(30))
random.shuffle(indices)
test = np.vstack([
        np.array(imgs_WhiteGlass)[indices[24:]],
        np.array(imgs_BrownGlass)[indices[24:]],
        np.array(imgs_GreenGlass)[indices[24:]]
    ])
labels_test = np.array([0] * 6 + [1] * 6 + [2] * 6)
text_labels = np.array(['WhiteGlass', 'BrownGlass', 'GreenGlass'])
result = loaded_model.score(test, labels_test)
print('Learning score: ' + str(result))


def predict_glass(img):
    img = find_edges(img)
    img = preprocess(img).flatten().reshape(1, -1)
    pred = loaded_model.predict_proba(img)  # reg.predict_proba(img)
    lab = text_labels[np.argmax(pred)]
    p = pred.max()
    return lab, p
