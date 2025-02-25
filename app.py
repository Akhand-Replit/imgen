import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image

# Set up the app title and icon
st.set_page_config(page_title="FLUX.1 Text to Image", page_icon="🌌")

# Initialize Hugging Face Inference client
def get_client():
    api_key = st.secrets.get("HUGGINGFACE_TOKEN")
    if not api_key:
        st.error("Hugging Face API token not found. Please check your secrets configuration.")
        st.stop()
    return InferenceClient(token=api_key)

client = get_client()

# App UI
st.title("🌌 FLUX.1 Text to Image Generator")
st.write("Transform text into images using the powerful FLUX.1 model!")

# Input parameters
prompt = st.text_input("Enter your prompt (e.g.: 'Cyberpunk samurai in neon rain')", 
                      placeholder="Describe the image you want to generate...")
negative_prompt = st.text_input("Negative prompt (optional)", 
                               placeholder="What to exclude from the image...")

col1, col2, col3 = st.columns(3)
with col1:
    guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
with col2:
    steps = st.slider("Number of Steps", 10, 150, 50)
with col3:
    height = st.selectbox("Height", [512, 768], index=0)
    width = st.selectbox("Width", [512, 768], index=0)

# Generate button
if st.button("Generate Image", type="primary"):
    if not prompt:
        st.warning("Please enter a prompt to generate an image")
        st.stop()
    
    with st.spinner("Generating your image... This may take 10-30 seconds"):
        try:
            image = client.text_to_image(
                prompt=prompt,
                model="black-forest-labs/FLUX.1-dev",  # Specific model added here
                negative_prompt=negative_prompt if negative_prompt else None,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=steps
            )
            
            st.image(image, caption=prompt, use_column_width=True)
            st.success("Image generated successfully!")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.stop()

# Add some app info
st.markdown("---")
st.markdown("""
**App Info:**
- Model: [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- Powered by [Hugging Face Inference API](https://huggingface.co/inference-api)
- Built with [Streamlit](https://streamlit.io)
""")
