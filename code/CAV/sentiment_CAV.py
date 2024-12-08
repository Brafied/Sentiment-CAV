import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression


data = np.load('../../results/short/last_token/short_review_activations_70B_quant.npy') # DATA
with open('../../results/short/synthetic_short_reviews_labels_70B_quant.pkl', 'rb') as f:
    labels = np.array(pickle.load(f))

model = LogisticRegression()
model.fit(data, labels)

coefficients = np.squeeze(model.coef_, axis=0).astype(np.float32)
intercept = np.squeeze(model.intercept_, axis=0).astype(np.float32)
np.savez('../../results/short/last_token/short_model_parameters_70B_quant.npz', coefficients=coefficients, intercept=intercept) # SAVE