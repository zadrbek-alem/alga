import numpy as np
from io import StringIO
import scipy
from scipy import ndimage
from sklearn.decomposition import PCA



def extract_features(images):
    wei = np.zeros((28, 28))
    for i in range(28):
        wei[i, i] = 69
    
    N = len(images)
        
    denoised_images = [ndimage.median_filter(img, size=3) for img in images]
    flattened = [img.flatten() for img in denoised_images]
    X = np.array(flattened)
    
    pca = PCA(n_components=10)
    features = pca.fit_transform(X)
    features_list = features.tolist()
    for i in range(len(features_list)):
        matrix = images[i]
        cur = scipy.ndimage.gaussian_filter(matrix, 4)
        cur2 = scipy.ndimage.laplace(cur)
        cur3 = scipy.ndimage.median_filter(cur, 5)
        cur4 = scipy.ndimage.sobel(matrix)
        cur5 = scipy.ndimage.prewitt(matrix)
        features_list[i].append(np.dot(matrix, wei).std())
        features_list[i].append(np.dot(matrix, wei).mean())
        features_list[i].append(np.dot(cur2, wei).std())
        features_list[i].append(np.dot(cur2, wei).mean())
        features_list[i].append(np.dot(cur3, wei).std())
        features_list[i].append(np.dot(cur3, wei).mean())
        features_list[i].append(np.dot(cur4, wei).std())
        features_list[i].append(np.dot(cur4, wei).mean())
        features_list[i].append(np.dot(cur5, wei).std())
        features_list[i].append(np.dot(cur5, wei).mean())


    return features_list


def get_total_input(text_lines):
    records = text_lines.strip().split("\n\n")
    N = int(records[0])
    records = records[1:]

    matrices = []
    for i in range(N):
        matrix = np.loadtxt(StringIO(records[i])).astype(int)
        matrices.append(matrix)
    return matrices


def float_list_to_str(float_list):
    string_list = [f"{num:.6f}" for num in float_list]
    return ' '.join(string_list)


def generate_total_output(matrices):
    features = extract_features(matrices)

    features_string_list = list(map(float_list_to_str, features))

    return '\n'.join(features_string_list)


input_txt = open("input.txt", "r").read()

matrices = get_total_input(input_txt)

feature_string = generate_total_output(matrices)

with open("output.txt", "w") as f:
    f.write(feature_string)
