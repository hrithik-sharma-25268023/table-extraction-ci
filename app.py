"""steamlit UI for Table Extraction"""

import os
import streamlit as st
from tqdm import tqdm

from src import pdf_image_utils, table_extraction

def main():
    """main method for app"""

    st.title("Table Extractor")

    file = st.file_uploader("Upload your PDF", type=["pdf"])

    if st.button("Extract") and file:
        os.makedirs("data/pdf", exist_ok=True)

        with open(f"data/pdf/{file.name}", "wb") as f:
            f.write(file.getbuffer())
        os.makedirs(f"data/images/{file.name}", exist_ok=True)
        os.makedirs(f"data/tables/{file.name}", exist_ok=True)

        pdf_image_utils.convert_pdf_to_image(source_path=f"data/pdf/{file.name}",
                                             destination_path=f"data/images/{file.name}")

        for pic in tqdm(os.listdir(f"data/images/{file.name}/")):
            image, results = table_extraction.table_detection(f"data/images/{file.name}/{pic}")
            table_extraction.save_results(image, results["scores"],
                                          results["labels"],
                                          results["boxes"],
                                          f"data/tables/{file.name}/{pic.replace('.jpg','')}.jpeg")

        st.success(f"All Tables Extracted.")

if __name__ == "__main__":
    main()
