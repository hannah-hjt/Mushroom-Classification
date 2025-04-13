
# 🍄 Mushroom Classification

**Mushroom Classification** is a machine learning project that identifies whether a mushroom is **poisonous** or **edible** based on various physical features. It uses the **Gaussian Naive Bayes** algorithm from `scikit-learn`, with performance evaluated using accuracy, precision, recall, and specificity.

---

## 📁 Dataset

The dataset is loaded from a CSV file named `CI-EX1.csv`, which includes categorical features describing mushrooms, such as:

- Cap shape, surface, and color
- Odor, bruises
- Gill spacing and color
- Stalk properties
- Ring number and type
- Spore print color
- Habitat, population
- ... and more

> The target variable is `class` — `e` for edible and `p` for poisonous.

---

## 🧠 Model

- **Algorithm:** Gaussian Naive Bayes (`GaussianNB`)
- **Library:** `scikit-learn`
- **Preprocessing:** One-hot encoding via `pandas.get_dummies()` for categorical features
- **Evaluation Metrics:** Accuracy, Precision, Recall, Specificity, Confusion Matrix

---

## 📊 Results & Visualization

The model is evaluated on both **validation** and **test** sets using:

- **Accuracy**
- **Precision**
- **Recall**
- **Specificity**

A confusion matrix is also visualized using `matplotlib`.

---

## 🧪 Data Splitting

The dataset is split as follows:

- **Training Set:** 70%
- **Validation Set:** 15% (half of remaining 30%)
- **Test Set:** 15% (other half)

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/mushroom-classification.git
cd mushroom-classification
pip install -r requirements.txt
```

---

## 🚀 Running the Code

Make sure `CI-EX1.csv` is placed in the project root. Then run:

```bash
python main.py
```

You'll see evaluation results printed in the terminal and a confusion matrix displayed.

---

## 📦 Requirements

```txt
pandas>=1.4.0
scikit-learn>=1.2.0
matplotlib>=3.5.0
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 📈 Example Output

```
Test Set:
Accuracity: 0.98
Precision: 0.97
Recall: 0.98
Specificity: 0.99

Validation Set:
Accuracity: 0.97
Precision: 0.96
Recall: 0.97
Specificity: 0.98
```

> ✅ Results will vary based on the dataset split and random state.

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
