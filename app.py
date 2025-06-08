import streamlit as st
import joblib 
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

#loading the saved pkl files
scaler = joblib.load('new_scaler.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')

# loading the segmented and cleaned data 
segmented_df = pd.read_csv('D:\\bank_segmentation_project\\customer_segments.csv')

# segment the action mapping 
segment_actions = {
    'Cash Advance Lovers' : 'send low interest offers to the customers',
    'Big Spenders' : 'offer a premium credit card with cashbacks',
    'Inactive Customers' : 'send reactivation promotions or surveys',
    'EMI Users' : 'Recommend buy low, pay later plans or EMIs'
}

# Page title 
st.title('Bank Customer Segmentation Dashboard')

# need to put some sidebars too! 
st.sidebar.title("segment filter")
segment_options = segmented_df['Segment'].dropna().unique()
selected_segment = st.sidebar.selectbox("choose a customer segment from the box", segment_options)

# filtered data 
filtered_data = segmented_df[segmented_df['Segment'] == selected_segment]

# now we want to display the segment insights 
st.subheader(f"segment: {selected_segment}")
st.markdown(f"recommended business action: {segment_actions.get(selected_segment, 'no action is mapped')}")
st.dataframe(filtered_data[['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT']].describe())

# plotting the pca plot
st.subheader("Cluster Visualization (PCA)")
features = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
            'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
            'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
            'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
            'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']
X_scaled = scaler.transform(segmented_df[features])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

viz_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
viz_df['Segment'] = segmented_df['Segment']

fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=viz_df, x='PCA1', y='PCA2', hue='Segment', palette='Set2', s=60, ax=ax)
st.pyplot(fig)

# then we need tpo download the data by converting the new data into a new csv file!
st.subheader("Download the segmented data")
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(segmented_df)
st.download_button(
    label="Download CSV",
    data = csv,
    file_name='customer_bank_segments.csv',
    mime='text/csv'
)

st.markdown("---")
st.markdown("Made by Somesh Rajendra Bhandarkar")