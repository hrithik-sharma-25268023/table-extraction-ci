"""steamlit UI for Table Extraction"""

import streamlit as st

def main():
    """main method for app"""

    st.title("Table Extractor")
    st.file_uploader("Upload you PDF:- ")
    
    if st.button('Extract'):
        st.text("Button Pressed")

if __name__ == "__main__":
    main()