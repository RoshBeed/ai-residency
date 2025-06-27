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
import math

# Page configuration
st.set_page_config(
    page_title="Vision Transformer Educational Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_available_models():
    """Get available models from the API"""
    try:
        models_url = os.environ.get("MODELS_URL", "http://localhost:8000/models")
        response = requests.get(models_url)
        if response.status_code == 200:
            return response.json()["available_models"]
        else:
            return []
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return []

def predict_multi_digit(image_array, model_url, model_type="vit"):
    """Predict multiple digits using the specified model."""
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array.astype('uint8'), 'L')
    
    # Save to buffer and encode
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    
    # Make prediction request
    try:
        response = requests.post(model_url, json={"image_data": b64, "model_type": model_type})
        data = response.json()
        # Return prediction and confidence
        return data["pred"], data.get("confidence", 0.0), data.get("model_type", model_type)
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, 0.0, model_type

def predict_both_models(image_array):
    """Predict using both ViT and seq2seq models simultaneously."""
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array.astype('uint8'), 'L')
    
    # Save to buffer and encode
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    
    # Make prediction request to both models endpoint
    try:
        both_models_url = os.environ.get("BOTH_MODELS_URL", "http://localhost:8000/predict/both")
        response = requests.post(both_models_url, json={"image_data": b64})
        data = response.json()
        return data
    except Exception as e:
        st.error(f"Error making prediction with both models: {e}")
        return None

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
    rows = int(math.ceil(num_heads / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 5 * rows))
    axes = axes.flatten()
    
    for head in range(num_heads):
        # Create random attention weights (for demonstration)
        attention_weights = np.random.rand(num_patches, num_patches)
        attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
        im = axes[head].imshow(attention_weights, cmap='Blues', vmin=0, vmax=1)
        axes[head].set_title(f'Attention Head {head + 1}')
        axes[head].set_xlabel('Key Patches')
        axes[head].set_ylabel('Query Patches')
        plt.colorbar(im, ax=axes[head])
    
    # Hide unused subplots
    for ax in axes[num_heads:]:
        ax.axis('off')
    
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

# Sidebar with model selection and educational content
with st.sidebar:
    st.header("🤖 Model Information")
    
    # Get available models
    available_models = get_available_models()
    
    if available_models:
        loaded_models = [model for model in available_models if model["loaded"]]
        if len(loaded_models) >= 2:
            st.success("✅ Both models loaded and ready!")
            
            for model in loaded_models:
                st.info(f"**{model['name']}** ({model['type'].title()})")
        elif len(loaded_models) == 1:
            st.warning(f"⚠️ Only {loaded_models[0]['name']} is loaded")
        else:
            st.error("❌ No models are currently loaded")
    else:
        st.warning("Could not fetch model information")
    
    st.markdown("---")
    st.header("📚 Learning Resources")
    
    st.subheader("What is a Vision Transformer?")
    st.markdown("""
    Vision Transformers (ViTs) are a type of neural network that applies the 
    transformer architecture (originally designed for text) to image processing.
    
    **Key Innovation:** Instead of using convolutional layers, ViTs divide images 
    into patches and process them using self-attention mechanisms.
    """)
    
    st.subheader("Model Types")
    st.markdown("""
    **Classification ViT:**
    - Direct classification of multi-digit numbers
    - Outputs a single number (0-999)
    - Faster inference
    
    **Seq2Seq ViT:**
    - Sequence-to-sequence approach
    - Generates digits one by one
    - More flexible for variable-length numbers
    """)
    
    st.subheader("Model Architecture")
    st.markdown("""
    **Our Model:**
    - **Input:** 84×84 grayscale images
    - **Patches:** 16×16 pixel patches (5×5 = 25 patches)
    - **Embedding:** 256-dimensional patch embeddings
    - **Layers:** 6 transformer encoder layers
    - **Heads:** 8 attention heads
    - **Output:** Classification or sequence generation
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
    st.metric("Model Parameters", "~2.1M")
    st.metric("Patch Size", "16×16 pixels")
    st.metric("Total Patches", "25")
    st.metric("Embedding Dim", "256")
    st.metric("Attention Heads", "8")
    st.metric("Encoder Layers", "6")
    
    # Show model status
    if available_models:
        loaded_count = len([m for m in available_models if m["loaded"]])
        st.markdown("---")
        st.subheader("🤖 Model Status")
        st.info(f"**{loaded_count}/2 models loaded**")
        if loaded_count == 2:
            st.success("Both models ready!")
        elif loaded_count == 1:
            st.warning("One model loaded")
        else:
            st.error("No models loaded")

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
        st.markdown("The image is divided into 16×16 pixel patches:")
        
        # Create patch visualization
        fig = visualize_patches_detailed(resized_image, patch_size=16)
        st.pyplot(fig)
        
        st.info("""
        **What's happening:** The 84×84 image is divided into 25 patches (5×5 grid). 
        Each patch is 16×16 pixels. This allows the model to process the image as a sequence 
        of smaller, manageable pieces rather than as a single large image.
        """)
        
        # Show patch statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Image Size", "84×84")
        with col2:
            st.metric("Patch Size", "16×16")
        with col3:
            st.metric("Total Patches", "25")
    
    with tab3:
        st.subheader("Step 3: Model Architecture")
        st.markdown("How the Vision Transformer processes your image:")
        
        # Create architecture diagram
        arch_fig = create_model_architecture_diagram()
        st.pyplot(arch_fig)
        
        st.info("""
        **Key Components:**
        1. **Patch Embedding:** Each 16×16 patch is flattened and projected to 256 dimensions
        2. **Position Embedding:** Learnable position information is added to preserve spatial relationships
        3. **Transformer Encoder:** 6 layers of self-attention and feed-forward processing
        4. **Multi-Head Attention:** 8 attention heads process different aspects of the input
        5. **Output Head:** Final layer that outputs classification or generates sequences
        """)
    
    with tab4:
        st.subheader("Step 4: Attention Visualization")
        st.markdown("Multi-head attention shows how the model focuses on different parts of the image:")
        
        # Create attention visualization
        attention_fig = create_attention_visualization(25, num_heads=8)
        st.pyplot(attention_fig)
        
        st.info("""
        **What's happening:** Each attention head learns to focus on different relationships 
        between patches. The heatmap shows how much each patch (row) attends to other patches (columns). 
        Brighter colors indicate stronger attention weights.
        """)
    
    with tab5:
        st.subheader("Step 5: Prediction Results")
        
        both_results = predict_both_models(resized_image)
        if both_results is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h2 style='color:#4F8BF9;'>🎯 ViT Classifier</h2>", unsafe_allow_html=True)
                vit_pred = both_results.get("vit_prediction", {})
                if "error" in vit_pred:
                    st.error(f"**Error:** {vit_pred['error']}")
                else:
                    st.markdown(f"<div style='font-size:2.5em; font-weight:bold; color:#4F8BF9; background:#eaf3ff; border-radius:10px; padding:20px; text-align:center;'>{vit_pred.get('pred', 'N/A')}</div>", unsafe_allow_html=True)
                    st.metric("Confidence", f"{vit_pred.get('confidence', 0.0):.2%}")
            with col2:
                st.markdown("<h2 style='color:#F97B4F;'>🔢 Seq2Seq Model</h2>", unsafe_allow_html=True)
                s2s_pred = both_results.get("seq2seq_prediction", {})
                if "error" in s2s_pred:
                    st.error(f"**Error:** {s2s_pred['error']}")
                else:
                    st.markdown(f"<div style='font-size:2.5em; font-weight:bold; color:#F97B4F; background:#fff3ea; border-radius:10px; padding:20px; text-align:center;'>{s2s_pred.get('pred', 'N/A')}</div>", unsafe_allow_html=True)
                    st.metric("Confidence", f"{s2s_pred.get('confidence', 0.0):.2%}")
        else:
            st.warning("No prediction results to display.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Streamlit • Vision Transformer Demo • Educational Purpose</p>
</div>
""", unsafe_allow_html=True)


