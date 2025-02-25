import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io
from datetime import datetime

# Set up the app title and icon
st.set_page_config(page_title="FLUX.1 Image Generator", page_icon="🌌")

# Initialize Hugging Face Inference client
def get_client():
    api_key = st.secrets.get("HUGGINGFACE_TOKEN")
    if not api_key:
        st.error("API token not found. Check secrets configuration.")
        st.stop()
    return InferenceClient(token=api_key)

client = get_client()

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# App UI
st.title("🌌 FLUX.1 Text to Image Generator")
st.write("Generate images using FLUX.1 model with advanced controls")

# Input parameters
with st.expander("⚙️ Generation Settings", expanded=True):
    prompt = st.text_input("Enter your prompt", 
                         placeholder="Describe the image you want to generate...")
    
    negative_prompt = st.text_input("Negative prompt (optional)", 
                                  placeholder="What to exclude from the image...")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
    with col2:
        steps = st.slider("Number of Steps", 10, 150, 50)
    with col3:
        num_images = st.slider("Number of Images", 1, 4, 1)
    with col4:
        height = st.selectbox("Height", [512, 768])
        width = st.selectbox("Width", [512, 768])

# Generate button
if st.button("Generate Images", type="primary"):
    if not prompt:
        st.warning("Please enter a prompt to generate images")
        st.stop()
    
    with st.spinner(f"Generating {num_images} image(s). This may take 20-60 seconds..."):
        try:
            images = client.text_to_image(
                prompt=prompt,
                model="black-forest-labs/FLUX.1-dev",
                negative_prompt=negative_prompt if negative_prompt else None,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=steps,
                num_images_per_prompt=num_images
            )
            
            # Handle single image response
            if not isinstance(images, list):
                images = [images]
            
            # Display generated images
            st.subheader("Generated Images")
            cols = st.columns(len(images))
            for idx, (col, image) in enumerate(zip(cols, images)):
                with col:
                    st.image(image, caption=f"Image {idx+1}", use_container_width=True)
                    
                    # Create download button
                    buf = io.BytesIO()
                    Image.open(io.BytesIO(image)).save(buf, format="PNG")
                    st.download_button(
                        label="Download",
                        data=buf.getvalue(),
                        file_name=f"flux_{prompt[:20]}_{idx+1}.png",
                        mime="image/png",
                        key=f"download_{idx}"
                    )
            
            # Add to history (store last 5 generations)
            st.session_state.history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prompt": prompt,
                "images": images,
                "params": {
                    "guidance_scale": guidance_scale,
                    "steps": steps,
                    "size": f"{width}x{height}"
                }
            })
            
            # Keep only last 5 generations
            if len(st.session_state.history) > 5:
                st.session_state.history.pop(0)
            
            st.success("Generation complete!")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()

# Display history
if st.session_state.history:
    st.markdown("---")
    st.subheader("Generation History")
    
    for gen in reversed(st.session_state.history):
        with st.expander(f"🕒 {gen['timestamp']} - {gen['prompt'][:50]}..."):
            st.write(f"**Prompt:** {gen['prompt']}")
            if gen['negative_prompt']:
                st.write(f"**Negative Prompt:** {gen['negative_prompt']}")
            st.write(f"**Parameters:** Guidance {gen['params']['guidance_scale']}, Steps {gen['params']['steps']}, Size {gen['params']['size']}")
            
            hist_cols = st.columns(len(gen['images']))
            for idx, (col, img) in enumerate(zip(hist_cols, gen['images'])):
                with col:
                    st.image(img, use_container_width=True)
                    buf = io.BytesIO()
                    Image.open(io.BytesIO(img)).save(buf, format="PNG")
                    st.download_button(
                        label="Download",
                        data=buf.getvalue(),
                        file_name=f"hist_{gen['timestamp']}_{idx+1}.png",
                        mime="image/png",
                        key=f"hist_dl_{gen['timestamp']}_{idx}"
                    )

# App info
st.markdown("---")
st.markdown("""
**App Info:**
- Model: [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- Images are stored temporarily in browser session
- Maximum 5 generations stored in history
- Built with Streamlit & Hugging Face Inference API
""")
