import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression


data = np.load('../../results/short/all_tokens/short_review_activations_70B_quant.npy', allow_pickle=True) # DATA
with open('../../results/short/synthetic_short_reviews_labels_70B_quant.pkl', 'rb') as f:
    labels = np.array(pickle.load(f))

expanded_data = []
expanded_labels = []
for i in range(len(data)):
    for j in range(len(data[i])):
        expanded_data.append(data[i][j])
        expanded_labels.append(labels[i])
expanded_data = np.array(expanded_data)
expanded_labels = np.array(expanded_labels)

model = LogisticRegression(max_iter=1000)
model.fit(expanded_data, expanded_labels)

coefficients = np.squeeze(model.coef_, axis=0).astype(np.float32)
intercept = np.squeeze(model.intercept_, axis=0).astype(np.float32)
np.savez('../../results/short/all_tokens/short_model_parameters_70B_quant.npz', coefficients=coefficients, intercept=intercept) # SAVE