# Customer Churn Modelling and Prediction Using Deep Learning and MLOps

In this project I am using Artificial Neural Network to model whether the person is going to leave the bank or not. It includes hyperparameter optimization (Optuna), experiment tracking (MLflow), and model export to ONNX format for portable inference.

The technologies used -  
    -	PyTorch Lightning —> High-level training loop.  
	-	Optuna —> Automated hyperparameter optimization.  
	-	MLflow —> Experiment tracking and model management.  
	-	ONNX —> Export format for interoperable inference.  
	-	scikit-learn —> Preprocessing utilities.   



## Data

I have `ChurnModelling.csv` file containf the required data for creating a ANN model.

**RowNumber**: An index representing the row number in the dataset. It is a sequential identifier for each record.

**CustomerId**: A unique identifier assigned to each customer. This ID can be used to distinguish individual customer records.

**Surname**: The last name of the customer. This is a textual field containing the surname of the individual.

**CreditScore**: A numerical value representing the credit score of the customer. Credit scores are used to evaluate the creditworthiness of an individual.

**Geography**: The country of residence of the customer. This field indicates the geographical location associated with the customer.

**Gender**: The gender of the customer, typically categorized as Male or Female.

**Age**: The age of the customer in years. This is a numerical field indicating how old the customer is.

**Tenure**: The number of years the customer has been with the bank. This field represents the duration of the customer's relationship with the bank.

**Balance**: The account balance of the customer. This is a numerical field that could represent the amount of money in the customer's account.

**NumOfProducts**: The number of bank products the customer uses, such as savings accounts, credit cards, etc. This is a numerical field indicating product usage.

**HasCrCard**: A binary indicator (1 or 0) showing whether the customer has a credit card. 1 means the customer has a credit card, and 0 means they do not.

**IsActiveMember**: A binary indicator (1 or 0) showing whether the customer is an active member. 1 indicates an active member, and 0 indicates an inactive member.

**EstimatedSalary**: An estimate of the customer's salary. This is a numerical field representing the estimated annual income of the customer.

**Exited**: A binary indicator (1 or 0) used to predict customer churn. 1 means the customer has left the bank, and 0 means the customer has not left the bank.


## Project Workflow

1. Load and Prepare Data  
	-	Input dataset: Churn_Modelling.csv   
	-	Drop unused identifiers (RowNumber, CustomerId, Surname)  
	-	Encode:  
	    -	Gender: Label Encoding  
	    -	Geography: One-Hot Encoding  
	-	Split the dataset into train, validation, test sets.


2. Preprocessing  
	-	Scale features using StandardScaler.  
	-	Save scalers and encoders using pickle for reproducibility:  
        -	label_encoder.pkl  
        -	one_hot_encoder.pkl  
        -	scaler.pkl  


3. Define Model
	-	A 3-layer fully connected feedforward neural network using PyTorch Lightning:  
	    -	Input → FC1 (ReLU) → FC2 (ReLU) → Output (Sigmoid)  
	-	Implements training_step, validation_step, test_step with metrics:
	    -	Accuracy, Precision, Recall, F1 Score, AUROC


4. Hyperparameter Tuning  
	-	Use Optuna to search for optimal:  
        -	Hidden layer sizes.  
        -	Learning rate.   
        -	Batch size.   
	-	Objective: minimize validation loss. 
	-	Training for each trial:  
        -	10 epochs  
        -	Early stopping. 
        -	Model checkpointing. 


5. Model Training (Best Trial)
	-	Retrain using best hyperparameters from Optuna.  
	-	Train for 30 epochs with early stopping.  
	-	Save the final model in ONNX format:
	    -	churn_model_best.onnx


6. Model Export (ONNX)
	-	Export model using torch.onnx.export().  
	-	Dynamic axes enabled for flexible batch size.  
	-	Useful for production deployment with ONNX Runtime or other frameworks.  


7. MLflow Logging
	-	Automatically logs:
	-	Hyperparameters
	-	Validation loss and metrics
	-	Artifacts:
	    -	ONNX model
	    -	Scalers and encoders
	    -	Raw dataset
	-	MLflow UI can be accessed at: http://localhost:5001


## Run the Project
    -	Install dependencies:  
        ```bash
        pip install -r requirements.txt
        ```
    -	Run the training script:  
        ```bash
        python model.py
        ```
    -	Start MLflow UI:  
        ```bash
        mlflow server --host localhost --port 5001 
        ```
    -	Access MLflow at: http://localhost:5001'
    -   Kill MLflow process:
        ```bash
        ps aux | grep mlflow | grep -v grep | awk '{print $2}' | xargs kill
        ```


## MLflow UI Screenshots

<img width="1547" alt="image" src="https://github.com/user-attachments/assets/7a5a522a-87db-4ac7-b771-b52ccd24ecad" />

<img width="1691" alt="image" src="https://github.com/user-attachments/assets/0e0d7ab9-1605-4b9c-a36a-70d228c5dd8e" />

<img width="1672" alt="image" src="https://github.com/user-attachments/assets/3e2e6671-c961-42d5-9f9a-ec6c86f31fe7" />

<img width="1675" alt="image" src="https://github.com/user-attachments/assets/db205a60-b597-4815-a465-669bae6dfc30" />

<img width="1694" alt="image" src="https://github.com/user-attachments/assets/27dc970c-e874-4282-a072-46503b2d5495" />


## Streamlit App
    -	Install Streamlit:  
        ```bash
        pip install streamlit
        ```
    -	Run the Streamlit app:  
        ```bash
        streamlit run churn_app.py
        ```
    -	Access the app at: http://localhost:8501

## Streamlit Visulizations

<img width="1044" alt="image" src="https://github.com/user-attachments/assets/e1bd9a4c-4f76-4dd7-b59d-0a0c5f8d09e7" />

<img width="690" alt="image" src="https://github.com/user-attachments/assets/8162a4cb-cd67-41d8-ab2f-c8c05c08c157" />

<img width="770" alt="image" src="https://github.com/user-attachments/assets/63ff65ad-86d7-4e4e-b23f-04e33187f5d4" />


