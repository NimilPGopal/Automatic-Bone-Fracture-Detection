import streamlit as st
from pathlib import Path
from PIL import Image

from core.predictions import predict
from core.gradcam import generate_gradcam
from core.pdf_report import generate_pdf_report


# --------------------------------------------------
# Page Config
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

icon_path = BASE_DIR / "assets" / "icon.png"

st.set_page_config(
    page_title="Automatic Bone Fracture Detection",
    page_icon=str(icon_path),
    layout="centered"
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
with st.sidebar:

    st.title("Project Information")

    st.markdown("---")

    st.subheader("Model Architecture")
    st.write("DenseNet121 CNN")

    st.subheader("Dataset")
    st.write("MURA (Musculoskeletal Radiographs)")

    st.subheader("Supported Body Parts")
    st.write("- Elbow")
    st.write("- Hand")
    st.write("- Shoulder")

    st.markdown("---")

    st.subheader("Features")

    st.write("✅ Fracture Detection")
    st.write("✅ Body Part Classification")
    st.write("✅ Grad-CAM Visualization")
    st.write("✅ PDF Diagnostic Report")

    st.markdown("---")
    st.info("AI Models Ready")
    st.markdown("---")


    st.subheader("Recommended Image Types")

    st.write("✔ X-ray Images")
    st.write("✔ JPG / JPEG / PNG Format")
    st.write("✔ Clear Radiographs")

    st.markdown("---")

    st.warning(
        "This system is intended for research and educational purposes only."
    )
st.markdown(
    """
    <h1 style='text-align:center;  color:#00BFFF; '>
        Automatic Bone Fracture Detection System
    </h1>

    <p style='text-align:center; font-size:18px; color:lightgray;'>
        AI-powered fracture detection using DenseNet121 and Grad-CAM
    </p>
    """,
    unsafe_allow_html=True
)
# --------------------------------------------------
# Create folders
# --------------------------------------------------

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

REPORT_DIR = Path("outputs/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

st.markdown("<br>", unsafe_allow_html=True)
# --------------------------------------------------
# Upload image
# --------------------------------------------------
st.markdown("## Upload X-ray Image")

st.write(
    "Upload a musculoskeletal X-ray image for AI-powered fracture analysis."
)
st.markdown("<br>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Browse or drag an X-ray image:",
    type=["png", "jpg", "jpeg"]
)
# --------------------------------------------------
# ONLY RUN AFTER IMAGE UPLOAD
# --------------------------------------------------

if uploaded_file is not None:

    # Save uploaded image
    image_path = UPLOAD_DIR / uploaded_file.name

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display image
    img = Image.open(image_path)
    img.thumbnail((500, 500))
    col1, col2, col3 = st.columns([1,2,1])

    with col2:

        st.image(
            img,
            caption="Uploaded X-ray",
            width=250
        )

    st.markdown("---")

    # Predict button
    if st.button("Analyze X-ray", use_container_width=True):

        with st.spinner("Analyzing X-ray..."):

            try:

                status = st.empty()

                # -------------------------------
                # Body Part Prediction
                # -------------------------------

                status.info("Detecting body part...")

                bone_type, bone_conf = predict(
                    str(image_path),
                    "Parts",
                    return_confidence=True
                )

                # -------------------------------
                # Fracture Prediction
                # -------------------------------

                status.info("Analyzing fracture...")

                result, result_conf = predict(
                    str(image_path),
                    bone_type,
                    return_confidence=True
                )

                # -------------------------------
                # Generate GradCAM
                # -------------------------------

                status.info("Generating Grad-CAM visualization...")

                gradcam_path = generate_gradcam(
                    str(image_path),
                    bone_type
                )

                gradcam_img = Image.open(gradcam_path)

                # -------------------------------
                # Generate PDF
                # -------------------------------

                status.info("Preparing diagnostic report...")

                pdf_path = REPORT_DIR / "Bone_Fracture_Report.pdf"

                generate_pdf_report(
                    save_path=str(pdf_path),
                    original_image_path=str(image_path),
                    gradcam_image_path=str(gradcam_path),
                    body_part=bone_type,
                    body_conf=bone_conf,
                    fracture_result=result,
                    fracture_conf=result_conf,
                )

                status.success("Analysis complete!")

                st.markdown("<br>", unsafe_allow_html=True)
                # -------------------------------
                # Tabs
                # -------------------------------
                st.markdown("<br>", unsafe_allow_html=True)
                tab1, tab2, tab3 = st.tabs([
                    "Results",
                    "Visualization",
                    "Report"
                ])

                # =========================================================
                # TAB 1 — RESULTS
                # =========================================================

                with tab1:

                    st.markdown("## Analysis Results")

                    metric1, metric2, metric3 = st.columns(3)

                    metric1.metric(
                        "Body Part",
                        bone_type
                    )

                    metric2.metric(
                        "Body Confidence",
                        f"{bone_conf:.2f}%"
                    )

                    metric3.metric(
                        "Fracture Confidence",
                        f"{result_conf :.2f}%"
                    )

                    st.write("Prediction Confidence:")

                    st.progress(int(result_conf))

                    if result == "fractured":

                        st.error("Fracture Detected⚠️ ")

                    else:

                        st.success("Normal Bone✅")

                # =========================================================
                # TAB 2 — VISUALIZATION
                # =========================================================

                with tab2:

                    st.markdown("## Diagnostic Visualization")

                    col1, col2 = st.columns(2)

                    with col1:

                        st.image(
                            img,
                            caption="Original X-ray",
                            use_column_width=True
                        )

                    with col2:

                        st.image(
                            gradcam_img,
                            caption="Grad-CAM Visualization",
                            use_column_width=True
                        )

                    st.markdown("### AI Interpretation:")

                    if result == "fractured":

                        st.error(
                            """
                            The model identified abnormal patterns associated
                            with a possible fracture.

                            The Grad-CAM visualization highlights regions 
                            that influenced the prediction.
                            """
                        )

                    else:

                        st.success(
                            """
                            The model did not detect strong fracture-related
                            abnormalities.

                            The Grad-CAM visualization highlights regions 
                            that influenced the prediction.
                            """
                        )

                # =========================================================
                # TAB 3 — REPORT
                # =========================================================

                with tab3:

                    st.markdown("## Diagnostic Report")

                    st.write(
                        "Download the complete AI-generated fracture analysis report."
                    )

                    with open(pdf_path, "rb") as pdf_file:

                        st.download_button(
                            label="Download Diagnostic Report",
                            data=pdf_file,
                            file_name="Bone_Fracture_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

            except Exception as e:

                st.error(f"An error occurred during analysis: {str(e)}")

                st.exception(e)
                    
st.markdown("---")

st.markdown(
    """
    <div style='text-align:center; color:gray;'>
        Automatic Bone Fracture Detection System using Deep Learning and Grad-CAM Explainability
    </div>
    """,
    unsafe_allow_html=True
)

