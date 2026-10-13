import os
import requests
import streamlit as st
from datetime import date, time

st.set_page_config(
    page_title="AML Transaction Risk Detection",
    page_icon="🛡️",
    layout="wide",
)

API_URL = os.getenv("AML_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.title("🛡️ AML Transaction Risk Detection")
st.write(
    "Submit transaction details to your FastAPI model and review the returned prediction. "
    "This tool supports investigation; a model result is not proof of money laundering."
)

with st.sidebar:
    st.header("API connection")
    st.code(f"{API_URL}/predict")
    if st.button("Check API health", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/health", timeout=8)
            if response.ok:
                st.success("API is reachable.")
                st.json(response.json())
            else:
                st.error(f"Health check failed (HTTP {response.status_code}).")
                st.code(response.text)
        except requests.RequestException as exc:
            st.error("Could not connect to FastAPI. Keep Uvicorn running in another terminal.")
            st.caption(str(exc))

st.subheader("Transaction details")
st.caption("Enter values that match the categories and formats used in your training dataset.")

with st.form("aml_transaction_form"):
    left, right = st.columns(2)

    with left:
        transaction_date = st.date_input("Date", value=date.today())
        transaction_time = st.time_input("Time", value=time(10, 30))
        sender_account = st.text_input("Sender account", value="account_1001")
        receiver_account = st.text_input("Receiver account", value="account_2001")
        amount = st.number_input("Amount", min_value=0.0, value=5000.0, step=100.0)

    with right:
        payment_currency = st.text_input("Payment currency", value="USD")
        received_currency = st.text_input("Received currency", value="USD")
        sender_bank_location = st.text_input("Sender bank location", value="USA")
        receiver_bank_location = st.text_input("Receiver bank location", value="USA")
        payment_type = st.text_input("Payment type", value="Wire")
        laundering_type = st.text_input(
            "Laundering type",
            value="Unknown",
            help="Your current API requires this feature. Use a value supported by your training data.",
        )

    submitted = st.form_submit_button(
        "Predict transaction risk",
        type="primary",
        use_container_width=True,
    )

if submitted:
    required_text_fields = {
        "Sender account": sender_account,
        "Receiver account": receiver_account,
        "Payment currency": payment_currency,
        "Received currency": received_currency,
        "Sender bank location": sender_bank_location,
        "Receiver bank location": receiver_bank_location,
        "Payment type": payment_type,
        "Laundering type": laundering_type,
    }
    empty_fields = [name for name, value in required_text_fields.items() if not value.strip()]

    if empty_fields:
        st.error("Please fill in: " + ", ".join(empty_fields))
    else:
        payload = {
            "Time": transaction_time.strftime("%H:%M:%S"),
            "Date": transaction_date.isoformat(),
            "Sender_account": sender_account.strip(),
            "Receiver_account": receiver_account.strip(),
            "Amount": float(amount),
            "Payment_currency": payment_currency.strip(),
            "Received_currency": received_currency.strip(),
            "Sender_bank_location": sender_bank_location.strip(),
            "Receiver_bank_location": receiver_bank_location.strip(),
            "Payment_type": payment_type.strip(),
            "Laundering_type": laundering_type.strip(),
        }

        with st.expander("Request payload", expanded=False):
            st.json(payload)

        with st.spinner("Sending transaction to the AML prediction API..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=30,
                )

                if response.ok:
                    result = response.json()
                    st.subheader("Prediction result")
                    st.success("Prediction received successfully.")
                    prediction = result.get("prediction")
                    if isinstance(prediction, (int, float, str)):
                        st.metric("Model prediction", str(prediction))
                    st.json(result)
                    st.caption(
                        "Interpret the prediction according to the label mapping used when training your model."
                    )
                else:
                    st.error(f"API returned HTTP {response.status_code}.")
                    try:
                        st.json(response.json())
                    except ValueError:
                        st.code(response.text)
                    st.info(
                        "If the response mentions missing features or invalid categories, "
                        "check that the submitted field names and values match app/main.py and your training data."
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to FastAPI. Start it in a separate terminal from your project root:\n\n"
                    "python -m uvicorn app.main:app --reload"
                )
            except requests.exceptions.Timeout:
                st.error("The API request timed out. Check the Uvicorn terminal for errors.")
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")

st.divider()
st.caption(f"Backend API: {API_URL} | FastAPI docs: {API_URL}/docs")
