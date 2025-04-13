
ShroomSense is an AI-powered project designed to distinguish between poisonous and non-poisonous mushrooms based on their attributes. Leveraging a supervised learning approach, the model is trained using the Gaussian Naive Bayes (GaussianNB) algorithm from the scikit-learn library. The primary goal is to accurately classify mushrooms and potentially prevent accidental poisoning.

Features
✅ Clean preprocessing of mushroom data

✅ Dataset split into training, validation, and test sets

✅ Model training using Gaussian Naive Bayes

✅ Performance evaluation on unseen data

✅ Easy-to-understand and reproducible codebase

Dataset
The dataset consists of structured mushroom characteristics such as cap shape, color, odor, gill attachment, stalk surface, and more. It has been split as follows:

Training Set: Remaining data after test and validation split

Validation Set: 1219 samples

Test Set: 1219 samples

⚠️ The dataset should be preprocessed to ensure all features are numeric or encoded properly for GaussianNB.

Model
Algorithm: Gaussian Naive Bayes (GaussianNB)

Library: scikit-learn

python
Copy
Edit
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
Installation
bash
Copy
Edit
git clone https://github.com/your-username/ShroomSense.git
cd ShroomSense
pip install -r requirements.txt
Usage
Prepare your dataset and ensure it is encoded correctly.

Run the training script:

bash
Copy
Edit
python train.py
Evaluate on test and validation sets.

Results
The model's performance is evaluated using accuracy, precision, recall, and F1 score on both the test and validation sets.

📊 Evaluation metrics will be logged in the console/output.

Requirements
Python 3.7+

scikit-learn

pandas

numpy

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Future Improvements
Add more sophisticated models (e.g., Random Forest, SVM)

Implement hyperparameter tuning

Build a web interface for interactive predictions

Improve feature visualization

