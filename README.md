
**Ti-6Al-4V Materials Informatics Portal**

*Machine Learning Assisted Microstructure Analysis and Property Prediction* of Ti-6Al-4V Titanium Alloy

An interactive web application that directly predicts **Vickers Hardness** and **ASTM E112 Grain Size** from SEM micrographs using metallurgically relevant features and Random Forest models.

###  Training Pipeline

The models were trained using the script [`training_pipeline.py`](training_pipeline.py).

**Key Highlights from Training:**
- Dataset: 1,224 SEM images of Ti-6Al-4V
- Features: Intensity statistics, Shannon Entropy, Edge Density
- Models: Random Forest Regressor & Classifier
- Results: R² = 0.9102 | Accuracy = 91.43%

### 📊 Model Performance

**Vickers Hardness Regression (Random Forest)**
- **R² Score**: **0.9102**
- **Mean Absolute Error**: **7.83 HV**

**Grain Size Classification (Random Forest)**
 **Accuracy**: **91.43%**

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Coarse   | 0.94      | 0.94   | 0.94     | 104     |
| Fine     | 0.97      | 0.94   | 0.96     | 115     |
| Medium   | 0.60      | 0.69   | 0.64     | 26      |

<img width="1538" height="665" alt="image" src="https://github.com/user-attachments/assets/59b57999-13d4-4574-8752-ade2a3e1fc74" />
<img width="1077" height="475" alt="Screenshot 2026-06-26 153809" src="https://github.com/user-attachments/assets/4b211f4f-6f84-486b-b3a4-241b9c2074e5" />


 **Key Features**

- **Real-time SEM Analysis** with advanced preprocessing (Bilateral Filter + CLAHE + Canny Edge)
- **Vickers Hardness Prediction**
- **ASTM E112 Grain Size Classification** (Fine / Medium / Coarse)
- **Interactive Visualizations** (Original image vs Grain Boundary Map + Intensity Histogram)
- **Physically Interpretable** results aligned with **Hall-Petch Relationship**
- Clean, professional Streamlit UI 



### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Laya1214/ti6al4v-materials-informatics.git
cd ti6al4v-materials-informatics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py

```
**Technologies Used**

Python • Streamlit • OpenCV • Scikit-learn • Pandas • Matplotlib

*License*
This project is open-sourced under the MIT License.
Feel free to use, modify, and contribute!
