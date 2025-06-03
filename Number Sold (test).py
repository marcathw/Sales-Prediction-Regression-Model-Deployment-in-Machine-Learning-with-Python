import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

def load_pickle(file):
    with open(file, "rb") as f:
        model = pickle.load(f)
    return model

def prediction(model, sample):
    sample_df = pd.DataFrame([sample], columns = ["store", "product", "year", "month", "day"])
    result = model.predict(sample_df)
    return result[0]

if __name__ == "__main__":
    file = "Number Sold.pkl"
    model = load_pickle(file)

    sample = [0, 0, 2019, 1, 1]
    result = prediction(model, sample)
    print(f"Predicted number_sold: {result}")