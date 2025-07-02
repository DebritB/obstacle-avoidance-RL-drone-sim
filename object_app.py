import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pickle
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from ultralytics import YOLO
import numpy as np
import cv2
import yaml

# Set page config
st.set_page_config(
    page_title="ChaosRAG - Landmine Identification System",
    page_icon="⚠️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #feca57);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #000000;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: linear-gradient(90deg, #ee6b6b, #feca57);
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown("""
<div class="main-header">
    <h1>⚠️ ChaosRAG - Landmine Identification System</h1>
    <p>AI-Powered Image Classification & Knowledge Retrieval for EOD Operations</p>
</div>
""", unsafe_allow_html=True)

# Safety warning
st.markdown("""
<div class="danger-box">
    <h3>🚨 CRITICAL SAFETY NOTICE</h3>
    <p><strong>This system is for educational and training purposes only. Always follow proper EOD protocols and consult with certified EOD specialists before handling any suspected explosive devices.</strong></p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_classification_model():
    """Load the YOLO object detection model and class names"""
    try:
        # Load YOLO model
        model = YOLO('best_new.pt')  # Update path if needed
        # Load class names from data.yaml
        with open('data.yaml', 'r') as file:
            data = yaml.safe_load(file)
        classes = data['names']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return model, classes, device
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
        return None, None, None

@st.cache_resource
def load_rag_system():
    """Load the RAG system components"""
    try:
        # Load embeddings and retriever
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Try to load existing retriever
        retriever_path = r"retriever_Landmines.pkl"
        vector_store_path = "Landmines_faiss_index_ollama"
        
        if os.path.exists(retriever_path):
            with open(retriever_path, 'rb') as f:
                retriever = pickle.load(f)
        elif os.path.exists(vector_store_path):
            persisted_vectorstore = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
            retriever = persisted_vectorstore.as_retriever()
        else:
            st.error("No retriever or vector store found!")
            return None, None
        
        # Initialize LLM
        llm = Ollama(model="llama3")
        
        # Define custom prompt template
        template = """You are a specialized explosive ordnance disposal (EOD) expert and landmine identification specialist. 
Your role is to assist defense personnel in identifying landmines based on field descriptions and available intelligence or 
provide all known information based on the name/type of the landmine.

Context: {context}

Field Description/Question: {question}

Expert Analysis:"""

        prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )

        # Create RetrievalQA with custom prompt
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return qa, retriever
        
    except Exception as e:
        st.error(f"Error loading RAG system: {str(e)}")
        return None, None

def classify_image(image, model, classes, device):
    """Detect objects in the uploaded image and return top 5 unique classes by highest confidence"""
    try:
        # Convert PIL image to OpenCV format
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        # Perform inference
        results = model(image_cv)
        result = results[0]
        boxes = result.boxes
        labels = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        # Apply confidence threshold
        threshold = 0.3
        filtered_indices = confs > threshold
        filtered_labels = labels[filtered_indices]
        filtered_confs = confs[filtered_indices]
        # Aggregate by class: keep highest confidence per class
        class_conf_dict = {}
        for label, conf in zip(filtered_labels, filtered_confs):
            if label not in class_conf_dict or conf > class_conf_dict[label]:
                class_conf_dict[label] = conf
        # Sort by confidence and get top 5 unique classes
        sorted_items = sorted(class_conf_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_labels = [item[0] for item in sorted_items]
        top5_confs = [item[1] for item in sorted_items]
        top5_classes = [classes[label] for label in top5_labels]
        top5_probs = top5_confs
        return top5_classes, top5_probs
    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
        return None, None

def get_landmine_info(landmine_name, qa_system):
    """Get detailed information about a landmine using RAG"""
    try:
        result = qa_system(landmine_name)
        return result["result"], result.get("source_documents", [])
    except Exception as e:
        st.error(f"Error retrieving landmine information: {str(e)}")
        return None, None

# Load models
model, classes, device = load_classification_model()
qa_system, retriever = load_rag_system()

# Main application layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Object Detection")
    st.markdown("Upload an image of a suspected landmine for AI-powered identification.")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear image of the suspected landmine"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        if model is not None and classes is not None:
            with st.spinner("Analyzing image..."):
                # Detection and summary
                top5_classes, top5_probs = classify_image(image, model, classes, device)

                # --- Draw bounding boxes for all detections ---
                image_cv = np.array(image)
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
                results = model(image_cv)
                result = results[0]
                boxes = result.boxes
                labels = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                threshold = 0.3
                filtered_indices = confs > threshold
                filtered_boxes = boxes.xyxy.cpu().numpy()[filtered_indices]
                filtered_labels = labels[filtered_indices]
                filtered_confs = confs[filtered_indices]
                for box, label, conf in zip(filtered_boxes, filtered_labels, filtered_confs):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    class_name = classes[label]
                    cv2.putText(image_cv, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                st.image(image_rgb, caption="Detection Results", use_column_width=True)
        else:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if top5_classes:
            st.markdown("### 🎯 Top 5 Predictions:")
            
            # Create a selectbox for the predictions
            prediction_options = []
            for i, (class_name, prob) in enumerate(zip(top5_classes, top5_probs)):
                prediction_options.append(f"{class_name} ({prob:.2%} confidence)")
            
            selected_prediction = st.selectbox(
                "Select a classification result for detailed information:",
                prediction_options,
                help="Choose from the top 5 predictions to get detailed information"
            )
            
            # Extract the class name from the selected option
            selected_class = selected_prediction.split(" (")[0]
            
            # Display classification results
            st.markdown("#### Classification Results:")
            for i, (class_name, prob) in enumerate(zip(top5_classes, top5_probs)):
                confidence_color = "🟢" if prob > 0.7 else "🟡" if prob > 0.4 else "🔴"
                st.markdown(f"{i+1}. **{class_name}** - {confidence_color} {prob:.2%}")

with col2:
    st.header("🧠 Knowledge Retrieval")
    st.markdown("Get detailed information about landmines using our AI knowledge base.")
    
    # Manual query option
    manual_query = st.text_input(
        "Or enter a landmine name/description manually:",
        placeholder="e.g., PMK-40, TM-62M, etc.",
        help="Enter the name or description of a landmine"
    )
    
    # Determine which query to use
    query_to_use = None
    if uploaded_file is not None and 'selected_class' in locals():
        query_to_use = selected_class
        st.info(f"Using classification result: **{selected_class}**")
    elif manual_query:
        query_to_use = manual_query
        st.info(f"Using manual query: **{manual_query}**")
    
    # Get landmine information
    if query_to_use and qa_system:
        with st.spinner("Retrieving information..."):
            info, sources = get_landmine_info(query_to_use, qa_system)
        
        if info:
            st.markdown("### 📋 Expert Analysis:")
            st.markdown(f"""
            <div class="result-box">
                {info}
            </div>
            """, unsafe_allow_html=True)
            
            # Show source documents
            if sources:
                with st.expander("📚 Source Documents"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(f"Page: {doc.metadata.get('page', 'Unknown')}")
                        st.markdown(f"Content: {doc.page_content[:300]}...")
                        st.markdown("---")

# Sidebar with additional information
with st.sidebar:
    st.header("ℹ️ System Information")
    
    st.markdown("### Model Status:")
    if model is not None:
        st.success("✅ Classification Model Loaded")
    else:
        st.error("❌ Classification Model Failed to Load")
    
    if qa_system is not None:
        st.success("✅ RAG System Loaded")
    else:
        st.error("❌ RAG System Failed to Load")
    
    device_info = "CUDA" if torch.cuda.is_available() else "CPU"
    st.info(f"🖥️ Running on: {device_info}")
    
    st.markdown("### 📖 Usage Instructions:")
    st.markdown("""
    1. **Upload Image**: Upload a clear image of the suspected landmine
    2. **Review Predictions**: Check the top 5 classification results
    3. **Select Prediction**: Choose a prediction from the dropdown
    4. **Get Information**: Detailed information will appear automatically
    5. **Manual Query**: You can also search manually by name
    """)
    
    st.markdown("### ⚠️ Safety Reminders:")
    st.markdown("""
    - Never approach suspected explosives
    - Always contact EOD specialists
    - This system is for training purposes only
    - Follow all safety protocols
    """)
    
    st.markdown("### 🔧 Supported Landmine Types:")
    if classes:
        st.markdown(f"The system can identify {len(classes)} different landmine types.")
        with st.expander("View all supported types"):
            for class_name in sorted(classes):
                st.markdown(f"• {class_name}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>ChaosRAG Landmine Identification System | For Educational and Training Purposes Only</p>
    <p>⚠️ Always consult with certified EOD specialists for real-world operations</p>
</div>
""", unsafe_allow_html=True) 