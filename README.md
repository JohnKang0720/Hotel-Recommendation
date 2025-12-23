# 🏨 Hotel Recommendation System

This project implements and compares **two collaborative filtering–based hotel recommendation systems**:

1. **Item-Based Collaborative Filtering**
2. **Singular Value Decomposition (SVD)–Based Collaborative Filtering**

The objective is to recommend hotels to users based on historical interaction data and to analyze the trade-offs between neighborhood-based and latent-factor-based recommendation approaches.

---

## 📌 Project Overview

Recommender systems are widely used in platforms such as Airbnb, Booking.com, and Expedia to personalize user experiences.  
This project explores:

- Item similarity–based recommendations
- Latent factor modeling using matrix factorization
- Differences in interpretability, scalability, and personalization

---

## 🧠 Recommendation Models

### 1️⃣ Item-Based Collaborative Filtering

**Notebook:**  
`hotel_recommendation (item-based).ipynb`

#### Methodology
- Constructs a **user–hotel interaction matrix**
- Computes **hotel-to-hotel similarity** using cosine similarity
- Recommends hotels similar to those previously interacted with by a user

#### Advantages
- Simple and intuitive
- Highly interpretable
- No model training required

#### Limitations
- Sensitive to sparse interaction data
- Cannot capture latent user preferences

---

### 2️⃣ SVD-Based Collaborative Filtering

**Notebook:**  
`hotel_recommendation.ipynb`

#### Methodology
- Applies **Singular Value Decomposition (SVD)** to the user–hotel matrix
- Decomposes interactions into latent user and item factors
- Predicts unseen interactions by reconstructing the matrix

#### Advantages
- Captures hidden user preferences
- Performs well on sparse datasets
- Produces more personalized recommendations

#### Limitations
- Less interpretable
- Requires hyperparameter tuning
- Cold-start problem for new users and items

---

## 📂 Repository Structure

```
Hotel-Recommendation/
│
├── hotel_recommendation.ipynb
├── hotel_recommendation (item-based).ipynb
└── README.md
```

## 🚀 How to Run

1. Clone the repository:
   ```bash git clone https://github.com/JohnKang0720/Hotel-Recommendation.git
   cd Hotel-Recommendation
    ```

2. Install Dependencies:
   ```
   pip install pandas numpy scikit-learn
   ```

4. Run the Notebook:
   ```
   jupyter notebook ...
   ```

