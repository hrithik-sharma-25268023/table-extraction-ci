"""steamlit UI for Table Extraction"""

import os
import streamlit as st
from src import pdf_image_utils

def main():
    """main method for app"""

    st.title("Table Extractor")

    file = st.file_uploader("Upload your PDF", type=["pdf"])

    if st.button("Extract") and file:
        os.makedirs("data/pdf", exist_ok=True)

        with open(f"data/pdf/{file.name}", "wb") as f:
            f.write(file.getbuffer())
        os.makedirs(f"data/images/{file.name}", exist_ok=True)

        pdf_image_utils.convert_pdf_to_image(source_path=f"data/pdf/{file.name}",
                                             destination_path=f"data/images/{file.name}")


if __name__ == "__main__":
    main()
