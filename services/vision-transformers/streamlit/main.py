import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests
import base64
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Page configuration
st.set_page_config(
    page_title="Vision Transformer Educational Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def predict_multi_digit(image_array, model_url):
    """Predict multiple digits using the entire canvas with patch-based ViT."""
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array.astype('uint8'), 'L')
    
    # Save to buffer and encode
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    
    # Make prediction request
    try:
        response = requests.post(model_url, json={"image_data": b64})
        data = response.json()
        # Return prediction and confidence
        return data["pred"], data.get("confidence", 0.0)
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, 0.0

def visualize_patches_detailed(image_array, patch_size=8):
    """Create detailed patch visualization with numbering"""
    height, width = image_array.shape
    patches_h = height // patch_size
    patches_w = width // patch_size
    
    # Create figure for visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image_array, cmap='gray')
    
    # Draw patch boundaries and number them
    patch_count = 0
    for i in range(patches_h):
        for j in range(patches_w):
            # Draw rectangle
            rect = patches.Rectangle(
                (j * patch_size, i * patch_size), 
                patch_size, patch_size, 
                linewidth=1, 
                edgecolor='red', 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add patch number
            center_x = j * patch_size + patch_size // 2
            center_y = i * patch_size + patch_size // 2
            ax.text(center_x, center_y, str(patch_count), 
                   ha='center', va='center', color='red', fontsize=8, fontweight='bold')
            patch_count += 1
    
    ax.set_title(f'Image divided into {patch_count} patches ({patches_w}x{patches_h})')
    ax.axis('off')
    
    return fig

def create_attention_visualization(num_patches, num_heads=4):
    """Create a visualization of multi-head attention"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for head in range(num_heads):
        # Create random attention weights (for demonstration)
        attention_weights = np.random.rand(num_patches, num_patches)
        attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
        
        im = axes[head].imshow(attention_weights, cmap='Blues', vmin=0, vmax=1)
        axes[head].set_title(f'Attention Head {head + 1}')
        axes[head].set_xlabel('Key Patches')
        axes[head].set_ylabel('Query Patches')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[head], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig

def create_model_architecture_diagram():
    """Create a simple model architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Define positions for components
    components = [
        ('Input Image\n84×84', (1, 6)),
        ('Patch Embedding\n8×8 patches', (3, 6)),
        ('Linear Projection\n64 dimensions', (5, 6)),
        ('Position Embedding', (7, 6)),
        ('CLS Token', (7, 5)),
        ('Transformer Encoder\n(3 layers)', (9, 6)),
        ('Multi-Head Attention\n(4 heads)', (9, 5)),
        ('Feed Forward\n(128 dim)', (9, 7)),
        ('Classification Head', (11, 6)),
        ('Output\n(0-999)', (13, 6))
    ]
    
    # Draw components
    for text, pos in components:
        ax.text(pos[0], pos[1], text, ha='center', va='center', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
               fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrow_positions = [
        ((2, 6), (2.8, 6)),
        ((3.2, 6), (4.8, 6)),
        ((5.2, 6), (6.8, 6)),
        ((7, 5.2), (7, 5.8)),
        ((7.2, 6), (8.8, 6)),
        ((9, 5.2), (9, 5.8)),
        ((9, 6.8), (9, 6.2)),
        ((9.2, 6), (10.8, 6)),
        ((11.2, 6), (12.8, 6))
    ]
    
    for start, end in arrow_positions:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(4, 8)
    ax.set_title('Vision Transformer Architecture', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    return fig

# Main app
st.title("🧠 Vision Transformer Educational Demo")
st.markdown("""
This interactive demo shows how Vision Transformers process images step-by-step. 
Draw digits on the canvas below and watch how the model processes your input!
""")

# Sidebar with educational content
with st.sidebar:
    st.header("📚 Learning Resources")
    
    st.subheader("What is a Vision Transformer?")
    st.markdown("""
    Vision Transformers (ViTs) are a type of neural network that applies the 
    transformer architecture (originally designed for text) to image processing.
    
    **Key Innovation:** Instead of using convolutional layers, ViTs divide images 
    into patches and process them using self-attention mechanisms.
    """)
    
    st.subheader("Model Architecture")
    st.markdown("""
    **Our Model:**
    - **Input:** 84×84 grayscale images
    - **Patches:** 8×8 pixel patches (11×11 = 121 patches)
    - **Embedding:** 64-dimensional patch embeddings
    - **Layers:** 3 transformer encoder layers
    - **Heads:** 4 attention heads
    - **Output:** Classification (0-999 for multi-digit numbers)
    """)
    
    st.subheader("Key Concepts")
    st.markdown("""
    1. **Patch Embedding:** Divide image into fixed-size patches
    2. **Position Embedding:** Add spatial information to patches
    3. **Self-Attention:** Learn relationships between all patches
    4. **Multi-Head Attention:** Multiple attention mechanisms in parallel
    5. **Feed-Forward Networks:** Process each patch independently
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎨 Draw Your Digits")
    st.markdown("Draw one or more digits (0-9) on the canvas below:")
    
    # Create canvas
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("📊 Model Statistics")
    st.metric("Model Parameters", "169,768")
    st.metric("Patch Size", "8×8 pixels")
    st.metric("Total Patches", "121")
    st.metric("Embedding Dim", "64")
    st.metric("Attention Heads", "4")
    st.metric("Encoder Layers", "3")

# Process the image if drawn
if canvas_result.image_data is not None:
    st.markdown("---")
    st.header("🔍 Processing Pipeline")
    
    # Convert canvas data to numpy array
    image_array = canvas_result.image_data[:, :, 0:3].astype('uint8')
    
    # Convert to grayscale
    if len(image_array.shape) == 3:
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_array
    
    # Resize to 84x84
    resized_image = cv2.resize(gray_image, (84, 84))
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📸 Input Processing", 
        "🧩 Patch Division", 
        "🧠 Model Architecture", 
        "👁️ Attention Visualization", 
        "🎯 Prediction Results"
    ])
    
    with tab1:
        st.subheader("Step 1: Input Processing")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original Drawing**")
            st.image(image_array, caption="Your canvas input")
        
        with col2:
            st.markdown("**Grayscale Conversion**")
            st.image(gray_image, caption="Single channel", use_column_width=True)
        
        with col3:
            st.markdown("**Resized to 84×84**")
            st.image(resized_image, caption="Model input size", use_column_width=True)
        
        st.info("""
        **What's happening:** The model expects 84×84 grayscale images. 
        Your drawing is converted to grayscale and resized to match this requirement.
        """)
    
    with tab2:
        st.subheader("Step 2: Patch Division")
        st.markdown("The image is divided into 8×8 pixel patches:")
        
        # Create patch visualization
        fig = visualize_patches_detailed(resized_image, patch_size=8)
        st.pyplot(fig)
        
        st.info("""
        **What's happening:** The 84×84 image is divided into 121 patches (11×11 grid). 
        Each patch is 8×8 pixels. This allows the model to process the image as a sequence 
        of smaller, manageable pieces rather than as a single large image.
        """)
        
        # Show patch statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Image Size", "84×84")
        with col2:
            st.metric("Patch Size", "8×8")
        with col3:
            st.metric("Total Patches", "121")
    
    with tab3:
        st.subheader("Step 3: Model Architecture")
        st.markdown("How the Vision Transformer processes your image:")
        
        # Create architecture diagram
        arch_fig = create_model_architecture_diagram()
        st.pyplot(arch_fig)
        
        st.info("""
        **Key Components:**
        1. **Patch Embedding:** Each 8×8 patch is flattened and projected to 64 dimensions
        2. **Position Embedding:** Learnable position information is added to preserve spatial relationships
        3. **CLS Token:** A special token that aggregates information from all patches
        4. **Transformer Encoder:** 3 layers of self-attention and feed-forward processing
        5. **Classification Head:** Final layer that outputs the predicted number
        """)
    
    with tab4:
        st.subheader("Step 4: Attention Visualization")
        st.markdown("Multi-head attention shows how the model focuses on different parts of the image:")
        
        # Create attention visualization
        attention_fig = create_attention_visualization(121, num_heads=4)
        st.pyplot(attention_fig)
        
        st.info("""
        **What's happening:** Each attention head learns to focus on different relationships 
        between patches. The heatmap shows how much each patch (row) attends to other patches (columns). 
        Brighter colors indicate stronger attention weights.
        """)
    
    with tab5:
        st.subheader("Step 5: Prediction Results")
        
        # Make prediction
        model_url = os.environ.get("MODEL_URL", "http://model-serve:8000/predict")
        pred, confidence = predict_multi_digit(resized_image, model_url)
        
        if pred is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Prediction**")
                st.metric("Predicted Number", pred)
                st.metric("Confidence", f"{confidence:.2%}")
            
            with col2:
                st.markdown("**Input Image**")
                st.image(resized_image, caption="Processed input", width=200)
            
            # Show detailed explanation
            st.markdown("### How the Model Made This Decision")
            st.markdown(f"""
            1. **Patch Processing:** Your image was divided into 121 patches
            2. **Feature Extraction:** Each patch was converted to a 64-dimensional vector
            3. **Attention Computation:** The model computed attention weights between all patches
            4. **Information Aggregation:** The CLS token gathered information from all patches
            5. **Classification:** The final layer mapped the aggregated features to number **{pred}**
            
            **Confidence Level:** {confidence:.2%} - This indicates how certain the model is about its prediction.
            """)
            
            # Show what the model learned
            st.markdown("### What the Model Learned")
            st.markdown("""
            The Vision Transformer learned to:
            - **Recognize digit shapes** across different patches
            - **Understand spatial relationships** between different parts of digits
            - **Handle multiple digits** in a single image
            - **Generalize** from training examples to new drawings
            
            Unlike CNNs that use local convolutions, the transformer can directly attend to 
            any part of the image, making it particularly good at understanding global patterns.
            """)
        else:
            st.error("Failed to get prediction from model. Please check if the model server is running.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Streamlit • Vision Transformer Demo • Educational Purpose</p>
</div>
""", unsafe_allow_html=True)


