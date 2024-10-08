# import streamlit as st
# import streamlit.components.v1 as components
# from pymongo import MongoClient
# import hashlib
# import pickle
# import numpy as np
# import shap
# import matplotlib.pyplot as plt
# import pandas as pd

# def load_model():
#     with open('model.pkl', 'rb') as file:
#         return pickle.load(file)


# # Connect to MongoDB
# with open('data.pkl','rb') as file:
#     data = pickle.load(file)


# client = MongoClient("mongodb://localhost:27017")
# db = client["b_cancer"]
# users_collection = db["users"]
# if "button_clicked" not in st.session_state:
#     st.session_state.button_clicked = False


# def callback():
#     st.session_state.button_clicked = True


# # Function to hash passwords
# def hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()


# # Function to check if a user exists in the database
# def user_exists(username):
#     return users_collection.find_one({"username": username}) is not None


# # Function to authenticate a user
# def authenticate_user(username, password):
#     hashed_password = hash_password(password)
#     user = users_collection.find_one({"username": username, "password": hashed_password})
#     return user is not None


# # Streamlit App
# def main():
#     st.title("Breast Cancer Classification App")

#     # Navigation
#     pages = ["Log In", "Sign Up"]
#     choice = st.sidebar.selectbox("Select Page", pages)


#     if choice == "Log In":
#         login_page()
#     elif choice == "Sign Up":
#         signup_page()


# def login_page():
#     st.subheader("Log In")
#     username = st.text_input("Username:")
#     password = st.text_input("Password:", type="password")


#     if (st.button("Log In",on_click = callback) or st.session_state.button_clicked):
#         if authenticate_user(username, password):
#                 st.success(f"Welcome, {username}!")


#             # Breast Cancer Classification
#                 sliders = {}
#                 for col in data.columns:
#                     default_value = data[col].mean()
#                     sliders[col] = st.slider(f"Select {col} range", min_value=data[col].min(), max_value=data[col].max(), value=default_value)

#                 submitted = st.button(label ="Submit",on_click = callback)
#                 if submitted:                  
#                         rf = load_model()
#                         # Make the prediction
#                         input_array = np.array([float(sliders[col]) for col in data.columns])
#                         input_features = np.array(input_array).reshape(1, -1)
#                         prediction = rf.predict(input_features)
                        
#                         #Using SHAP 
#                         explainer = shap.TreeExplainer(rf)
#                         shap_values = explainer.shap_values(input_features)
#                         shap.initjs()
#                         shap.force_plot(explainer.expected_value[1],shap_values[1], feature_names=data.columns, matplotlib=True)
#                         st.pyplot(plt.gcf())
#                         from lime import lime_tabular
                        
#                         # Using LIME 
#                         explainer1 = lime_tabular.LimeTabularExplainer(data.values, feature_names=data.columns, class_names=['Benign', 'Malignant'], discretize_continuous=True)

#                         # Local Instance taken from input features
#                         instance_to_explain = pd.DataFrame(input_features, columns=data.columns)

#                         exp = explainer1.explain_instance(instance_to_explain.values[0], rf.predict_proba, num_features=len(data.columns))
#                         st.components.v1.html(exp.as_html(), width=800, height=1000)
                        
                        
#                         # Display the prediction
#                         st.subheader("Prediction:")
#                         if prediction[0] == 0:
#                             st.write("The tumor is predicted to be benign.")
#                         else:
#                             st.write("The tumor is predicted to be malignant.")

#         else:
#             st.error("Invalid username or password. Please try again.")


# # For Authentication a Sign Up Page 
# def signup_page():
#     st.subheader("Sign Up")
#     new_username = st.text_input("New Username:")
#     new_password = st.text_input("New Password:", type="password")
#     confirm_password = st.text_input("Confirm Password:", type="password")
#     signup_button = st.button("Sign Up", key="signup")

#     if signup_button:
#         if new_password == confirm_password:
#             if not user_exists(new_username):
#                 hashed_password = hash_password(new_password)
#                 users_collection.insert_one({"username": new_username, "password": hashed_password})
#                 st.success("Registration successful! You can now log in.")
#             else:
#                 st.error("Username already exists. Choose a different username.")
#         else:
#             st.error("Passwords do not match. Please try again.")

# if __name__ == "__main__":
#     main()
import streamlit as st
import streamlit.components.v1 as components
from pymongo import MongoClient
import hashlib
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
from lime import lime_tabular

def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

def shap_load_model():
    with open('shap.pkl', 'rb') as file:
        return pickle.load(file)

def lime_load_model():
    with open('lime.pkl', 'rb') as file:
        return pickle.load(file)


# Connect to MongoDB
with open('data.pkl','rb') as file:
    data = pickle.load(file)
with open('lime_features.pkl','rb') as file:
    lime_features = pickle.load(file)
with open('shap_features.pkl','rb') as file:
    shap_features = pickle.load(file)

client = MongoClient("mongodb://localhost:27017")
db = client["b_cancer"]
users_collection = db["users"]
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False


def callback():
    st.session_state.button_clicked = True


# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Function to check if a user exists in the database
def user_exists(username):
    return users_collection.find_one({"username": username}) is not None


# Function to authenticate a user
def authenticate_user(username, password):
    hashed_password = hash_password(password)
    user = users_collection.find_one({"username": username, "password": hashed_password})
    return user is not None


# Streamlit App
def main():
    st.title("Breast Cancer Classification App")

    # Navigation
    pages = ["Log In", "Sign Up"]
    choice = st.sidebar.selectbox("Select Page", pages)

    if choice == "Log In":
        login_page()
    elif choice == "Sign Up":
        signup_page()


def login_page():
    st.subheader("Log In")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")

    if (st.button("Log In", key="login_button", on_click=callback) or st.session_state.button_clicked):
        if authenticate_user(username, password):
            st.success(f"Welcome, {username}!")
            pages = ["SHAP", "LIME"]
            choice = st.selectbox("Select XAI", pages)

            if choice == "SHAP":
                shap_explanation()
            elif choice == "LIME":
                lime_explanation()
            # Breast Cancer Classification

        else:
            st.error("Invalid username or password. Please try again.")


# SHAP Explanation Function
def shap_explanation():
    sliders = {}
    for col in shap_features.columns:
        slider_key = f"{col}_slider_shap"
        default_value = float(shap_features[col].mean())
        sliders[col] = st.slider(f"Select {col} range", min_value=shap_features[col].min(), max_value=shap_features[col].max(), value=default_value, key=slider_key)

    submitted = st.button(label="Submit", key="shap_submit_button", on_click=callback)
    if submitted:
        rf = shap_load_model()
        # Make the prediction
        input_array = np.array([float(sliders[col]) for col in shap_features.columns])
        input_features = np.array(input_array).reshape(1, -1)
        prediction = rf.predict(input_features)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(input_features)
        shap.initjs()
        shap.force_plot(explainer.expected_value[0], shap_values[..., 0], matplotlib=True)
        st.pyplot(plt.gcf())
        st.subheader("Prediction:")
        if prediction[0] == 0:
            st.write("The tumor is predicted to be benign.")
        else:
            st.write("The tumor is predicted to be malignant.")


# LIME Explanation Function
def lime_explanation():
    sliders = {}
    for col in lime_features.columns:
        slider_key = f"{col}_slider_lime"
        default_value = float(lime_features[col].mean())
        sliders[col] = st.slider(f"Select {col} range", min_value=lime_features[col].min(), max_value=lime_features[col].max(), value=default_value, key=slider_key)

    submitted = st.button(label="Submit", key="lime_submit_button", on_click=callback)
    if submitted:
        rf = lime_load_model()
        # Make the prediction
        input_array = np.array([float(sliders[col]) for col in lime_features.columns])
        input_features = np.array(input_array).reshape(1, -1)
        prediction = rf.predict(input_features)
        explainer = lime_tabular.LimeTabularExplainer(lime_features.values, feature_names=lime_features.columns, class_names=['Benign', 'Malignant'], discretize_continuous=True)
        instance_to_explain = pd.DataFrame(input_features, columns=lime_features.columns)
        exp = explainer.explain_instance(instance_to_explain.values[0], rf.predict_proba, num_features=len(lime_features.columns))
        st.components.v1.html(exp.as_html(), width=800, height=1000)
        st.subheader("Prediction:")
        if prediction[0] == 0:
            st.write("The tumor is predicted to be benign.")
        else:
            st.write("The tumor is predicted to be malignant.")


# Sign Up Page
def signup_page():
    st.subheader("Sign Up")
    new_username = st.text_input("New Username:")
    new_password = st.text_input("New Password:", type="password")
    confirm_password = st.text_input("Confirm Password:", type="password")
    signup_button = st.button("Sign Up", key="signup_button", on_click=callback)

    if signup_button:
        if new_password == confirm_password:
            if not user_exists(new_username):
                hashed_password = hash_password(new_password)
                users_collection.insert_one({"username": new_username, "password": hashed_password})
                st.success("Registration successful! You can now log in.")
            else:
                st.error("Username already exists. Choose a different username.")
        else:
            st.error("Passwords do not match. Please try again.")


if __name__ == "__main__":
    main()

