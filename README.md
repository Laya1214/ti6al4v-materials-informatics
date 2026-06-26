
### Ti-6Al-4V Materials Informatics Portal

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


	
### Results

*Model Performance*

<img width="1077" height="475" alt="Screenshot 2026-06-26 153809" src="https://github.com/user-attachments/assets/4b211f4f-6f84-486b-b3a4-241b9c2074e5" />
<img width="1538" height="665" alt="image" src="https://github.com/user-attachments/assets/59b57999-13d4-4574-8752-ade2a3e1fc74" />


*Streamlit results*

*Fine grain Structure*
<img width="1910" height="869" alt="image" src="https://github.com/user-attachments/assets/5b974575-44ce-47b9-ae75-5cb5106067f4" />
<img width="1608" height="906" alt="image" src="https://github.com/user-attachments/assets/73cc0721-6136-4b59-b6b6-eb5867663bc8" />
<img width="1695" height="640" alt="image" src="https://github.com/user-attachments/assets/2e47bc7c-4ead-46dd-bf9c-fcac15ba3f2f" />
<img width="1913" height="881" alt="image" src="https://github.com/user-attachments/assets/ce0f75ce-a49f-4b6a-a70a-fcc507495392" />
<img width="1919" height="896" alt="image" src="https://github.com/user-attachments/assets/6c293b20-b7b6-47c4-adf8-4162988c795d" />
<img width="1838" height="729" alt="image" src="https://github.com/user-attachments/assets/f9cc624f-a0e2-44b4-be2f-45ef2443bcb6" />

*Coarse grain structure*

<img width="1912" height="894" alt="image" src="https://github.com/user-attachments/assets/6db326d5-9366-472a-8f5b-ed576fbe4bde" />
<img width="1885" height="883" alt="image" src="https://github.com/user-attachments/assets/dbb7579c-c530-447a-a4fc-c832ba626fcc" />
<img width="1891" height="834" alt="image" src="https://github.com/user-attachments/assets/64baa683-cc57-4fc0-8b09-940d63382860" />

*Medium grain structure*

<img width="1895" height="891" alt="image" src="https://github.com/user-attachments/assets/c1766fdd-aff0-4e7a-a9c9-c6d074e16cf4" />
<img width="1893" height="884" alt="image" src="https://github.com/user-attachments/assets/29b9cb30-db8c-463b-8f89-c9b5c3c6ca06" />
<img width="1909" height="835" alt="image" src="https://github.com/user-attachments/assets/fbdc3d7a-795d-4fd1-9c4f-9b806e8fee26" />


