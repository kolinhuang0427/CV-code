import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
from getImageFeatures import get_image_features
from getImageDistance import get_image_distance

def evaluate_nearest_neighbor(test_images, train_features, train_labels, dictionary, distance_method):
    predictions = []

    for img_name in test_images:
        # Load the word map
        word_map = pickle.load(open(f'../data/{img_name[:-4]}_{dictionary}.pkl', 'rb'))
        features = get_image_features(word_map, 500)  # Assuming get_image_features is defined
        
        # Calculate distances to all training features
        distances = [get_image_distance(features, train_features[i], method=distance_method) for i in range(train_features.shape[0])]
        
        # Find the nearest neighbor
        nearest_index = np.argmin(distances)
        predictions.append(train_labels[nearest_index])

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    
    # Confusion matrix
    C = confusion_matrix(test_labels, predictions)

    return accuracy, C

if __name__ == "__main__":
    # Load data
    meta = pickle.load(open('../data/traintest.pkl', 'rb'))
    train_image_names = meta['train_imagenames']
    train_labels = meta['train_labels']
    test_image_names = meta['test_imagenames']
    test_labels = meta['test_labels']

    # Load features and dictionaries
    vision_random = pickle.load(open('visionRandom.pkl', 'rb'))
    vision_harris = pickle.load(open('visionHarris.pkl', 'rb'))

    # Evaluate Random Dictionary
    accuracy_random_euclidean, confusion_random_euclidean = evaluate_nearest_neighbor(
        test_image_names, vision_random['trainFeatures'], vision_random['trainLabels'],
        'Random', 'euclidean'
    )
    print(f"Random Dictionary - Euclidean Distance: Accuracy = {accuracy_random_euclidean:.4f}")
    print(f"Confusion Matrix:\n{confusion_random_euclidean}")

    accuracy_random_chi2, confusion_random_chi2 = evaluate_nearest_neighbor(
        test_image_names, vision_random['trainFeatures'], vision_random['trainLabels'],
        'Random', 'chi2'
    )
    print(f"Random Dictionary - Chi-squared Distance: Accuracy = {accuracy_random_chi2:.4f}")
    print(f"Confusion Matrix:\n{confusion_random_chi2}")

    # Evaluate Harris Dictionary
    accuracy_harris_euclidean, confusion_harris_euclidean = evaluate_nearest_neighbor(
        test_image_names, vision_harris['trainFeatures'], vision_harris['trainLabels'],
        'Harris', 'euclidean'
    )
    print(f"Harris Dictionary - Euclidean Distance: Accuracy = {accuracy_harris_euclidean:.4f}")
    print(f"Confusion Matrix:\n{confusion_harris_euclidean}")

    accuracy_harris_chi2, confusion_harris_chi2 = evaluate_nearest_neighbor(
        test_image_names, vision_harris['trainFeatures'], vision_harris['trainLabels'],
        'Harris', 'chi2'
    )
    print(f"Harris Dictionary - Chi-squared Distance: Accuracy = {accuracy_harris_chi2:.4f}")
    print(f"Confusion Matrix:\n{confusion_harris_chi2}")
