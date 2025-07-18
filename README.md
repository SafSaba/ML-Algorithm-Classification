
# Machine Learning Classification Algorithms

This repository contains implementations of various machine learning classification algorithms to compare their performance on a given dataset.

## 📖 Overview

The primary goal of this project is to apply several common classification models to a dataset and evaluate their effectiveness. The models are trained and tested, and their performance is measured using standard classification metrics.

---

## 🤖 Algorithms Covered

This project explores the following classification algorithms:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest

---
## ⚙️ Setup with Anaconda

Follow these instructions to set up the project and its dependencies using the Anaconda distribution.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SafSaba/ML-Algorithm-Classification.git](https://github.com/SafSaba/ML-Algorithm-Classification.git)
    cd ML-Algorithm-Classification
    ```

2.  **Create the Conda Environment:**
    The `environment.yml` file in this repository contains all the necessary packages. You can create an identical environment with a single command.
    ```bash
    # Create the environment from the yml file
    conda env create -f environment.yml
    ```

3.  **Activate the Environment:**
    Once the environment is created, activate it to start using it. The name of the environment is specified inside the `environment.yml` file.
    ```bash
    # Activate the new environment
    conda activate <your-env-name>
    ```
    *(Note: Replace `<your-env-name>` with the actual name defined in your `environment.yml` file.)*

After activating the environment, you are ready to run the project's scripts or notebooks as described in the **Usage** section.
