# Fluid Analysis Reasoning Intelligent System

**Fluid Analysis Reasoning Intelligent System (FARIS)** is an intelligent pipeline for analyzing oil condition reports using a combination of cutting-edge AI techniques, including Computer Vision, Machine Learning, Deep Learning, and Natural Language Processing. It extracts key information from fluid analysis documents and classifies oil condition while providing transparent reasoning behind each classification decision.

---

## 🚀 Features

* 🔍 **OCR Extraction**: Automatically extracts text from scanned oil reports using PaddleOCR.
* 🌳 **Decision Tree Classification**: Determines the condition of the oil based on structured report data.
* 🧠 **Shallow Neural Network**: Adds robustness to the classification pipeline using a lightweight deep learning model.
* 🗣 **RAG-powered Reasoning**: Generates human-readable explanations for each classification using Retrieval-Augmented Generation (RAG) techniques.
* 📊 **Insightful Dashboard** *(WIP)*: Displays key findings and classification outcomes in an intuitive, visual format.

---

## 🛠️ Tech Stack

* **Language**: Python
* **Computer Vision**: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
* **Machine Learning**: Scikit-learn (Decision Trees)
* **Deep Learning**: PyTorch or Keras (Shallow Neural Network)
* **NLP**: Retrieval-Augmented Generation (RAG)
* **Visualization** *(planned)*: Matplotlib, Plotly, Streamlit

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/Mo-Sam-Mo/Fluid-Analysis-Reasoning-Intelligent-System.git
cd Fluid-Analysis-Reasoning-Intelligent-System

# 2. (Optional) Create a virtual environment
python -m venv venv
venv/Scripts/activate  # On Windows use `venv\Scripts\activate`

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start you app
streamlit run StreamlitDeploy.py
```

---


## 🧠 System Pipeline Overview

1. **OCR**: Convert image-based reports into structured text using PaddleOCR.
2. **Preprocessing**: Clean and extract relevant entities from the text.
3. **Classification**: Use a decision tree or shallow neural network to determine oil condition.
4. **Reasoning**: Generate classification rationale using a RAG-based NLP model.
5. **Visualization** *(Upcoming)*: Display results and insights in an interactive dashboard.

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## 👥 Contributors

* [Mohamed Samir](https://github.com/Mo-Sam-Mo)
* [Omar Mahmoud](https://github.com/OmarGira)
* [Ahmed Halim](https://github.com/Ahmed-M-Halim)
* [Roaa Shehab](https://github.com/Roaa1932002)
* [Abdelrahman Bakeer](https://github.com/A-Bakeer)

---

## 💡 Future Plans

* Add support for multilingual OCR (e.g., Arabic, French)
* Implement real-time processing capabilities
* Add more advanced reasoning using LLMs

---

## 📬 Contact

For questions or feedback, feel free to open an issue or contact the contributors directly.
