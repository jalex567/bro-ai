import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib  # Import joblib
import io
import base64
import os

# Set page config
st.set_page_config(
    page_title="Accident Severity Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 30px;}
    .sub-header {font-size: 1.8rem; color: #ff7f0e; margin-top: 30px; margin-bottom: 15px;}
    .feature-header {font-size: 1.3rem; color: #2ca02c; margin-top: 20px;}
    .stButton>button {width: 100%;}
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🚗 Accident Severity Prediction System</h1>', unsafe_allow_html=True)
st.write("This application predicts accident severity using a Random Forest model with class balancing.")

# Initialize session state for model persistence
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'clf' not in st.session_state:
    st.session_state.clf = None

# Sidebar for data upload and model management
st.sidebar.title("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your CSV data file", type=["csv"])

# Model management section
st.sidebar.title("Model Management")
randomforest_file = st.sidebar.file_uploader("Upload a trained model (joblib)", type=["joblib"])

# Function to load sample data if no file is uploaded
@st.cache_data
def load_sample_data():
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'Start_Time': pd.date_range('2020-01-01', periods=n_samples, freq='H'),
        'Distance(mi)': np.random.uniform(0, 5, n_samples),
        'Temperature(F)': np.random.uniform(20, 100, n_samples),
        'Humidity(%)': np.random.uniform(10, 100, n_samples),
        'Visibility(mi)': np.random.uniform(0, 10, n_samples),
        'Wind_Speed(mph)': np.random.uniform(0, 30, n_samples),
        'Weather_Condition': np.random.choice(['Clear', 'Rain', 'Snow', 'Fog', 'Cloudy'], n_samples),
        'Side': np.random.choice(['R', 'L'], n_samples),
        'Junction': np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
        'Traffic_Signal': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
        'Amenity': np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
        'Crossing': np.random.choice([True, False], n_samples, p=[0.15, 0.85]),
        'Severity': np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    })
    
    return sample_data

# Load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Data uploaded successfully!")
else:
    df = load_sample_data()
    st.sidebar.info("Using sample data for demonstration.")

# Load model if provided
if fandomforest_file is not None:
    try:
        st.session_state.clf = joblib.load(randomforest_file)
        st.session_state.model_trained = True
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")

# Show data preview
if st.sidebar.checkbox("Show Data Preview"):
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    st.subheader("Data Summary")
    st.write(f"Dataset shape: {df.shape}")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Numerical Columns:**")
        st.write(df.select_dtypes(include=[np.number]).columns.tolist())
    with col2:
        st.write("**Categorical Columns:**")
        st.write(df.select_dtypes(include=['object']).columns.tolist())

# -----------------------------
# Enhanced Feature engineering
# -----------------------------
st.markdown('<h2 class="sub-header">Feature Engineering</h2>', unsafe_allow_html=True)

df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['Hour'] = df['Start_Time'].dt.hour
df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
df['Month'] = df['Start_Time'].dt.month
df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df['Is_Night'] = ((df['Hour'] >= 20) | (df['Hour'] <= 6)).astype(int)

# -----------------------------
# Identify numeric and categorical features
# -----------------------------
possible_numeric_features = [
    'Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)',
    'Wind_Speed(mph)', 'Hour', 'DayOfWeek', 'Month', 'Is_Weekend', 'Is_Night'
]

possible_categorical_features = [
    'Weather_Condition', 'Side', 'Junction', 'Traffic_Signal',
    'Amenity', 'Crossing'
]

# Keep only features that exist in the dataset
numeric_features = [f for f in possible_numeric_features if f in df.columns]
categorical_features = [f for f in possible_categorical_features if f in df.columns]

# -----------------------------
# Prepare data
# -----------------------------
X = df[numeric_features + categorical_features]
y = df['Severity']

st.write("Class distribution in training data:")
st.write(y.value_counts().sort_index())

# -----------------------------
# Preprocessing pipeline
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# -----------------------------
# Calculate class weights
# -----------------------------
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))
st.write("Class weights:", class_weight_dict)

# -----------------------------
# Full pipeline with class weighting
# -----------------------------
if st.session_state.clf is None:
    st.session_state.clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight=class_weight_dict,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        ))
    ])

# -----------------------------
# Train/test split with stratification
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train the model
# -----------------------------
if st.button('Train Model'):
    with st.spinner('Training model with class balancing...'):
        st.session_state.clf.fit(X_train, y_train)
        st.session_state.model_trained = True
        st.success('Model training completed!')
        
        # Save model button
        if st.button('Save Model'):
            joblib.dump(st.session_state.clf, 'accident_severity_model.joblib')
            st.success('Model saved as accident_severity_model.joblib')
        
        # -----------------------------
        # Predictions and evaluation
        # -----------------------------
        y_pred = st.session_state.clf.predict(X_test)
        
        st.markdown('<h2 class="sub-header">Model Evaluation</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Classification Report:**")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
        
        with col2:
            st.write("**Confusion Matrix:**")
            cm = confusion_matrix(y_test, y_pred)
            st.write(cm)
            
            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=np.unique(y), yticklabels=np.unique(y),
                        cbar_kws={'label': 'Fraction of Actual Class'}, ax=ax)
            ax.set_title('Normalized Confusion Matrix')
            ax.set_ylabel('Actual Severity')
            ax.set_xlabel('Predicted Severity')
            st.pyplot(fig)
        
        # -----------------------------
        # Enhanced Visualization Functions
        # -----------------------------
        st.markdown('<h2 class="sub-header">Visualizations</h2>', unsafe_allow_html=True)
        
        classes = sorted(np.unique(y))
        
        # Plot class distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        train_counts = y_train.value_counts().sort_index()
        ax1.bar(train_counts.index.astype(str), train_counts.values, color='skyblue', alpha=0.7)
        ax1.set_title('Training Set Class Distribution')
        ax1.set_xlabel('Severity Class')
        ax1.set_ylabel('Count')
        
        test_counts = y_test.value_counts().sort_index()
        ax2.bar(test_counts.index.astype(str), test_counts.values, color='lightgreen', alpha=0.7)
        ax2.set_title('Test Set Class Distribution')
        ax2.set_xlabel('Severity Class')
        ax2.set_ylabel('Count')
        st.pyplot(fig)
        
        # Plot classification metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics = ['precision', 'recall', 'f1-score']
        data = {metric: [] for metric in metrics}
        
        for cls in classes:
            cls_str = str(cls)
            if cls_str in report and cls_str != 'accuracy':
                for metric in metrics:
                    data[metric].append(report[cls_str][metric])
            else:
                for metric in metrics:
                    data[metric].append(0)
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#3498db', '#2ecc71', '#e74c3c']  # Blue, Green, Red
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, data[metric], width, label=metric.capitalize(),
                   color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Severity Class')
        ax.set_ylabel('Score')
        ax.set_title('Classification Metrics by Class')
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value annotations
        for i, cls in enumerate(classes):
            for j, metric in enumerate(metrics):
                height = data[metric][i]
                if height > 0.01:
                    ax.text(i + j*width, height + 0.02, f'{height:.2f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        st.pyplot(fig)
        
        # -----------------------------
        # Additional analysis
        # -----------------------------
        st.markdown('<h2 class="sub-header">Additional Analysis</h2>', unsafe_allow_html=True)
        
        unique_preds = np.unique(y_pred)
        st.write(f"Unique classes predicted: {unique_preds}")
        
        missing_classes = set(classes) - set(unique_preds)
        if missing_classes:
            st.write(f"Classes not predicted at all: {missing_classes}")
        else:
            st.write("All classes were predicted at least once!")
        
        st.write("Sample counts per class:")
        for cls in classes:
            count = (y_test == cls).sum()
            st.write(f"Class {cls}: {count} samples")
        
        # -----------------------------
        # Feature importance
        # -----------------------------
        try:
            # Get feature importance from the model
            importances = st.session_state.clf.named_steps['classifier'].feature_importances_
            
            # Get feature names after preprocessing
            numeric_features = st.session_state.clf.named_steps['preprocessor'].transformers_[0][2]
            categorical_features = st.session_state.clf.named_steps['preprocessor'].transformers_[1][2]
            
            # Get one-hot encoded feature names
            ohe = st.session_state.clf.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
            categorical_feature_names = ohe.get_feature_names_out(categorical_features)
            
            # Combine all feature names
            all_feature_names = numeric_features.tolist() + categorical_feature_names.tolist()
            
            # Create a DataFrame for feature importance
            feature_imp = pd.DataFrame({
                'feature': all_feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(15)
            
            # Plot feature importance
            st.markdown('<h3 class="feature-header">Feature Importance</h3>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=feature_imp, ax=ax)
            ax.set_title('Top 15 Most Important Features')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not plot feature importance: {e}")

# -----------------------------
# Prediction interface
# -----------------------------
st.markdown('<h2 class="sub-header">Make Predictions</h2>', unsafe_allow_html=True)

if st.session_state.model_trained:
    # Create input form for prediction
    with st.form("prediction_form"):
        st.write("Enter feature values for prediction:")
        
        col1, col2 = st.columns(2)
        
        input_data = {}
        with col1:
            for feature in numeric_features:
                if feature != 'Severity':
                    # Set default values based on sample data statistics
                    if feature in df.columns:
                        default_val = df[feature].median() if df[feature].dtype != 'object' else df[feature].mode()[0]
                        input_data[feature] = st.number_input(feature, value=float(default_val))
        
        with col2:
            for feature in categorical_features:
                if feature in df.columns:
                    options = df[feature].unique().tolist()
                    default_val = df[feature].mode()[0] if len(options) > 0 else None
                    input_data[feature] = st.selectbox(feature, options, index=options.index(default_val) if default_val in options else 0)
        
        submitted = st.form_submit_button("Predict Severity")
        
        if submitted:
            # Create input DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            try:
                prediction = st.session_state.clf.predict(input_df)
                st.success(f"Predicted Severity: {prediction[0]}")
                
                # Show prediction probabilities
                probabilities = st.session_state.clf.predict_proba(input_df)
                prob_df = pd.DataFrame({
                    'Severity': st.session_state.clf.classes_,
                    'Probability': probabilities[0]
                }).sort_values('Probability', ascending=False)
                
                st.write("Prediction probabilities:")
                st.dataframe(prob_df)
                
                # Plot probabilities
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Severity', y='Probability', data=prob_df, ax=ax)
                ax.set_title('Prediction Probabilities by Severity Class')
                ax.set_ylim(0, 1)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
else:
    st.info("Please train the model first to make predictions.")
