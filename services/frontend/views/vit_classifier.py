"""Vision Transformer: patch embedding to class."""

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from client import call
from components import draw_patch_grid, render_probability_bars
from shared import bytes_to_b64, canvas_to_png_bytes, service_page, show

SERVICE = "vit-classifier"


def render() -> None:
    with service_page(SERVICE):
        col_draw, col_upload = st.columns(2)
        with col_draw:
            st.caption("Draw a digit")
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=18,
                stroke_color="#ffffff",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="vit_classifier_canvas",
            )
        with col_upload:
            st.caption("Or upload one")
            uploaded = st.file_uploader(
                "Upload a digit image (PNG/JPG)",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
            )
            if uploaded is not None:
                st.image(uploaded, width=150)

        image_bytes: bytes | None = None
        if canvas_result.image_data is not None:
            image_bytes = canvas_to_png_bytes(canvas_result.image_data)
        elif uploaded is not None:
            image_bytes = uploaded.getvalue()

        if image_bytes is not None and st.button("Classify", type="primary"):
            b64_image = bytes_to_b64(image_bytes)
            vit_data = show(call(SERVICE, "/predict", {"image_data": b64_image}))

            if vit_data:
                st.caption(
                    f"Input: {vit_data['img_size']}x{vit_data['img_size']}px, "
                    f"{vit_data['patch_size']}x{vit_data['patch_size']} patches."
                )
                st.image(
                    draw_patch_grid(image_bytes, vit_data["img_size"], vit_data["patch_size"]),
                    width=260,
                )
                st.markdown(f"**Predicted: {vit_data['pred']}**")
                if vit_data.get("class_probabilities"):
                    st.markdown(
                        render_probability_bars(vit_data["class_probabilities"], vit_data["pred"]),
                        unsafe_allow_html=True,
                    )
