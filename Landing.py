import streamlit as st

st.set_page_config(page_title="Subgroup 6 - Landing Page", page_icon="🌐")

st.title("👋 Welcome to Subgroup 6 Project Portal")
st.markdown("## Choose an Application to Get Started")

# Buttons to go to each page
col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/app.py", label="📊 Loan Default Risk Assessment")

with col2:
    st.page_link("pages/spam_app.py", label="🛡️ Spam Detection in Network Traffic")

# Footer
st.markdown("---")
st.markdown("### 👥 Subgroup 6 members")
st.markdown("Sambit Bhoumik R-2262079")
st.markdown("Sukanya Mondal R-2262078")
st.markdown("Kaushal Kumar R-2262082")
st.markdown("Rakesh Mondal R-2262083")
st.markdown("Arka Mitra R-2262084")

