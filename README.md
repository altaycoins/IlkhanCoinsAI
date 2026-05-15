# 🪙 altaycoins Ilkhan Ruler Identification Tool

Welcome to the **altaycoins Ilkhan Ruler Identification Tool**! This is a Streamlit-based web application that utilizes a deep learning model (EfficientNetV2) to identify the ruler of Ilkhanid coins from images. 

## ✨ Features

* **AI-Powered Identification:** Uses a fine-tuned EfficientNetV2 model to predict the ruler associated with a given coin.
* **Automatic Background Removal:** Integrates `rembg` to automatically strip backgrounds from uploaded coin images before processing, improving prediction accuracy.
* **Smart Image Processing:** Automatically splits the coin image into obverse and reverse halves, processes them as separate tensors, and averages the predictions for a highly confident final result.
* **User-Friendly Interface:** Built with Streamlit for a clean, intuitive drag-and-drop web experience.
* **Top 3 Guesses:** Displays the most likely ruler along with the next best guesses and their confidence percentages.

## 🛠️ Prerequisites

Before you begin, ensure you have Python 3.8+ installed on your system. 

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/altaycoins/altaycoinsAI.git](https://github.com/altaycoins/altaycoinsAI.git)
cd altaycoinsAI