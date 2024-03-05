
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
sns.set(style='dark')

all_df = pd.read_csv("main_data.csv")

datetime_columns = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date", "review_creation_date", "review_answer_timestamp", "shipping_limit_date"]
for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

with st.sidebar:
  st.image("https://raw.githubusercontent.com/reanpies/analisis_data_dengan_python/c878cd9044b8124e16c995fa64d4dc316c9bfb74/profilWA.png")

  start_date, end_date = st.date_input(
    "Rentang Waktu",
    min_value=all_df["order_purchase_timestamp"].min(),
    max_value=all_df["order_purchase_timestamp"].max(),
    value=(all_df["order_purchase_timestamp"].min(),
    all_df["order_purchase_timestamp"].max())
  )

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

main_df = all_df[(all_df["order_purchase_timestamp"] >= start_date) & (all_df["order_purchase_timestamp"] <= end_date)]

daily_orders_df = main_df.resample(rule='D', on='order_purchase_timestamp').agg({
    "order_id": "nunique",
    "price": "sum"
}).reset_index()
daily_orders_df.columns = ["order_date", "order_count", "revenue"]

sum_order_items_df = main_df.groupby("product_category_name")["order_id"].count().sort_values(ascending=False).reset_index()

bystate_df = main_df.groupby("customer_state")["customer_id"].nunique().reset_index()
bystate_df.columns = ["state", "customer_count"]

bycity_df = main_df.groupby("customer_city")["customer_id"].nunique().reset_index()
bycity_df.columns = ["city", "customer_count"]

rfm_df = main_df.groupby("customer_unique_id").agg({
    "order_purchase_timestamp": "max",
    "order_id": "nunique",
    "price": "sum"
}).reset_index()
rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
recent_date = main_df["order_purchase_timestamp"].dt.date.max()
rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
rfm_df.drop("max_order_timestamp", axis=1, inplace=True)

st.title("Bangkit Collection Dashboard 🔥")

st.header('Daily Orders')
st.line_chart(daily_orders_df.set_index("order_date"))

st.header('Product Performance')
st.subheader('Best Performing Products')
st.bar_chart(sum_order_items_df.head(10).set_index("product_category_name"))

st.subheader('Worst Performing Products')
st.bar_chart(sum_order_items_df.tail(10).set_index("product_category_name"))

st.header('Customer Demographics')
st.subheader('Number of Customers by State')
st.bar_chart(bystate_df.set_index("state"))
st.subheader('Number of Customers by City')
st.bar_chart(bycity_df.set_index("city"))

st.header('Best Customer Based on RFM Parameters')
st.metric("Average Recency (days)", value=round(rfm_df.recency.mean(), 1))
st.metric("Average Frequency", value=round(rfm_df.frequency.mean(), 2))
st.metric("Average Monetary", value=format_currency(rfm_df.monetary.mean(), "AUD ", locale='es_CO'))

st.subheader('Top 5 Customers by Recency')
st.bar_chart(rfm_df.sort_values(by="recency").head(5).set_index("customer_id"))

st.subheader('Top 5 Customers by Frequency')
st.bar_chart(rfm_df.sort_values(by="frequency", ascending=False).head(5).set_index("customer_id"))

st.subheader('Top 5 Customers by Monetary')
st.bar_chart(rfm_df.sort_values(by="monetary", ascending=False).head(5).set_index("customer_id"))


st.subheader("Product Category Distribution")

monthly_orders_df = all_df.resample(rule='M', on='order_purchase_timestamp').agg({
    "order_id": "nunique",
    "price": "sum",
    "product_category_name": lambda x: x.mode().iat[0] if not x.mode().empty else None
})

monthly_orders_df.index = monthly_orders_df.index.strftime('%B')
monthly_orders_df = monthly_orders_df.reset_index()
monthly_orders_df.rename(columns={
    "order_purchase_timestamp": "waktu_pesanan",
    "order_id": "jumlah_pesanan",
    "price": "penghasilan",
    "product_category_name": "produk_terlaris"
}, inplace=True)

canceled_orders_df = all_df[all_df['order_status'] == 'canceled']

canceled_products_count = canceled_orders_df['product_category_name'].value_counts()

fig, ax = plt.subplots(figsize=(20, 10))
colors = sns.color_palette("Dark2", n_colors=len(monthly_orders_df))
sns.barplot(x="jumlah_pesanan", y="produk_terlaris", data=monthly_orders_df, palette=colors, ax=ax)
ax.set_title("Monthly Orders by Product Category", fontsize=30)
ax.set_xlabel("Number of Orders", fontsize=20)
ax.set_ylabel("Product Category", fontsize=20)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(15, 8))
sns.barplot(x=canceled_products_count.head(10).values, y=canceled_products_count.head(10).index, color='skyblue', ax=ax)
ax.set_title("Top 10 Canceled Product Categories", fontsize=25)
ax.set_xlabel("Number of Cancellations", fontsize=15)
ax.set_ylabel("Product Category", fontsize=15)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
st.pyplot(fig)


st.subheader("Customer Behavior Analysis")

review_cancel_prob = all_df.groupby('review_score')['order_status'].apply(lambda x: (x == 'canceled').mean())
delivery_time_cancel_prob = all_df.groupby(pd.cut(all_df['order_delivery_time'], bins=[0, 24, 48, 72, float('inf')]))['order_status'].apply(lambda x: (x == 'canceled').mean())
purchase_history_cancel_prob = all_df.groupby('customer_unique_id')['order_status'].apply(lambda x: (x == 'canceled').mean())
top_categories = all_df['product_category_name'].value_counts().head(10).index
category_cancel_prob = all_df[all_df['product_category_name'].isin(top_categories)].groupby('product_category_name')['order_status'].apply(lambda x: (x == 'canceled').mean())

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=review_cancel_prob.index, y=review_cancel_prob.values, color='skyblue', ax=ax)
ax.set_title("Review Score vs Cancel Probability", fontsize=20)
ax.set_xlabel("Review Score", fontsize=15)
ax.set_ylabel("Cancel Probability", fontsize=15)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=delivery_time_cancel_prob.index.astype(str), y=delivery_time_cancel_prob.values, color='skyblue', ax=ax)
ax.set_title("Delivery Time vs Cancel Probability", fontsize=20)
ax.set_xlabel("Delivery Time (hours)", fontsize=15)
ax.set_ylabel("Cancel Probability", fontsize=15)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(purchase_history_cancel_prob.values, bins=20, kde=False, color='skyblue', ax=ax)
ax.set_title("Purchase History vs Cancel Probability", fontsize=20)
ax.set_xlabel("Cancel Probability", fontsize=15)
ax.set_ylabel("Count", fontsize=15)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
st.pyplot(fig)


st.subheader("Top Categories Cancel Probability")

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=category_cancel_prob.values, y=category_cancel_prob.index, color='skyblue', ax=ax)
ax.set_title("Cancel Probability by Top Categories", fontsize=20)
ax.set_xlabel("Cancel Probability", fontsize=15)
ax.set_ylabel("Product Category", fontsize=15)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
st.pyplot(fig)

st.caption('Terimakasih Bangkit Academy 2024')
st.caption('Copyright © Rayhan Faiz M009D4KY2393')
