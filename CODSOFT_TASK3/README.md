# Customer Churn Prediction

CodSoft Machine Learning Internship · Task 3

A random forest predicts the `Exited` field from ten account and profile features. The pipeline one-hot encodes geography and gender and excludes identifiers and surnames. The app uses a 50% decision threshold.

## Run on Windows

Extract the project under `D:\CodSoft`, open PowerShell in `CODSOFT_TASK3`, and paste:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_data.py
.\.venv\Scripts\python.exe app.py
```

Open [http://127.0.0.1:8003](http://127.0.0.1:8003). Keep PowerShell open while using the app. The first page load trains the model. Restarting the app retrains it.

If the automatic download is unavailable, download and extract the [assignment dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction). Place the following files directly in the project folder's `data` subfolder: `data/Churn_Modelling.csv`. Create the subfolder if needed.

On Linux or macOS, use `python3 -m venv .venv` and replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

## Demo

Click **Load example** to populate a fictional customer profile, then **Predict churn**. Change one input and run another prediction. Click the theme button at the top to switch between dark and light modes. The browser remembers the selection.

## Evaluation

A stratified 80/20 split preserves the churn ratio. Preprocessing and training use the training split. The app reports accuracy, churn F1, and churn recall on the held-out split. Recall measures how many customers who actually left were flagged.

The model is an internship demonstration. It includes demographic features from the dataset and has no fairness or calibration assessment. Use fictional profiles during the demo.

## Project files

- `app.py` serves the app and prediction API.
- `model.py` trains, evaluates, and runs the model.
- `index.html` contains the responsive interface and theme controls.
- `setup_data.py` downloads the assignment dataset.
- `requirements.txt` lists Python dependencies.

The repository excludes the downloaded dataset and local environment. Follow the setup instructions to reproduce training and measurements.
