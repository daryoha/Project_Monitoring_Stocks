from google.colab import drive
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

def read_data():
    drive.mount('/content/gdrive/')
    data = pd.read_csv("/content/gdrive/MyDrive/fin_quotes_per_day.csv")
    return data

def get_company_data(data, company):
    data = data.set_index("Date")
    data = data[[company]]
    data.index = pd.to_datetime(data.index)
    return data

def split_train_test_data(data):
    train_data, test_data = train_test_split(data, test_size=0.25, shuffle=False)
    return train_data, test_data

def get_Xy_train_test(train_data, test_data, company):
    train = train_data.reset_index()[company]
    test = test_data.reset_index()[company]
    train = np.array(train).reshape(-1, 1)
    test = np.array(test).reshape(-1, 1)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    # scaling dataset
    scaled_train = scaler.fit_transform(train)
    # Normalizing values between 0 and 1
    scaled_test = scaler.fit_transform(test)
    # Create sequences and labels for training data
    sequence_length = 50  # Number of time steps to look back
    X_train, y_train = [], []
    for i in range(len(scaled_train) - sequence_length):
        X_train.append(scaled_train[i:i+sequence_length])
        y_train.append(scaled_train[i+1:i+sequence_length+1])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    # Create sequences and labels for testing data
    sequence_length = 7  # Number of time steps to look back
    X_test, y_test = [], []
    for i in range(len(scaled_test) - sequence_length):
        X_test.append(scaled_test[i:i+sequence_length])
        y_test.append(scaled_test[i+1:i+sequence_length+1])
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # Convert data to PyTorch tensors
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    return X_train, y_train, X_test, y_test, scaler

class LSTMModel(nn.Module):
      # input_size : number of features in input at each time step
      # hidden_size : Number of LSTM units 
      # num_layers : number of LSTM layers 
    def __init__(self, input_size, hidden_size, num_layers): 
        super(LSTMModel, self).__init__() #initializes the parent class nn.Module
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
 
    def forward(self, x): # defines forward pass of the neural network
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out

def train(X_train, y_train, X_test, y_test, company):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 1
    num_layers = 2
    hidden_size = 64
    output_size = 1
    
    # Define the model, loss function, and optimizer
    model = LSTMModel(input_size, hidden_size, num_layers).to(device)
    
    loss_fn = torch.nn.MSELoss(reduction='mean')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 16
    # Create DataLoader for batch training
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create DataLoader for batch training
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    num_epochs = 50
    train_hist =[]
    test_hist =[]
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0

        # Training
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate average training loss and accuracy
        average_loss = total_loss / len(train_loader)
        train_hist.append(average_loss)

        # Validation on test data
        model.eval()
        with torch.no_grad():
            total_test_loss = 0.0

            for batch_X_test, batch_y_test in test_loader:
                batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
                predictions_test = model(batch_X_test)
                test_loss = loss_fn(predictions_test, batch_y_test)

                total_test_loss += test_loss.item()

            # Calculate average test loss and accuracy
            average_test_loss = total_test_loss / len(test_loader)
            test_hist.append(average_test_loss)
        if (epoch+1)%10==0:
            print(f'Epoch [{epoch+1}/{num_epochs}] - Training Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}')

    x = np.linspace(1,num_epochs,num_epochs)
    plt.plot(x,train_hist,scalex=True, label="Training loss")
    plt.plot(x, test_hist, label="Test loss")
    plt.legend()
    plt.show()

    PATH = f"/content/Project_Monitoring_Stocks/dl_models/{company}_model.path"
    torch.save(model.state_dict(), PATH)
    
def predcit(model, test_data, X_test, scaler, device, company):
    # Define the number of future time steps to forecast
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_forecast_steps = 7

    # Convert to NumPy and remove singleton dimensions
    sequence_to_plot = X_test.squeeze().cpu().numpy()

    # Use the last 30 data points as the starting point
    historical_data = sequence_to_plot[-1]

    # Initialize a list to store the forecasted values
    forecasted_values = []

    # Use the trained model to forecast future values
    with torch.no_grad():
        for _ in range(num_forecast_steps*2):
            # Prepare the historical_data tensor
            historical_data_tensor = torch.as_tensor(historical_data).view(1, -1, 1).float().to(device)
            # Use the model to predict the next value
            predicted_value = model(historical_data_tensor).cpu().numpy()[0, 0]

            # Append the predicted value to the forecasted_values list
            forecasted_values.append(predicted_value[0])

            # Update the historical_data sequence by removing the oldest value and adding the predicted value
            historical_data = np.roll(historical_data, shift=-1)
            historical_data[-1] = predicted_value

    last_date = test_data.index[-1]
    # Generate the next 30 dates
    future_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=7)

    # Concatenate the original index with the future dates
    combined_index = test_data.index.append(future_dates)


    forecasted_cases = scaler.inverse_transform(np.expand_dims(forecasted_values, axis=0)).flatten() 

    prediction_result = {}

    prediction_result["Date"] = list(test_data.index[-100:-7]) + list(combined_index[-7:])
    prediction_result[company] = list(test_data[company][-100:-7]) + list(forecasted_cases)
    
    return prediction_result