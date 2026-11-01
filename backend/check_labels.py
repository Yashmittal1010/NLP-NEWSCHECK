import pandas as pd

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\fake-news-detection\data\raw\WELFake_Dataset.csv")

reuters_rows = df[df["text"].str.contains(r"WASHINGTON \(Reuters\)", na=False, regex=True)]
print("Rows matching 'WASHINGTON (Reuters)' dateline:", len(reuters_rows))
print("\nLabel distribution for those rows:")
print(reuters_rows["label"].value_counts())

print("\n--- For comparison, a few other outlet-style checks ---")

breitbart_rows = df[df["text"].str.contains("Breitbart", na=False, case=False)]
print("\nRows mentioning 'Breitbart':", len(breitbart_rows))
print(breitbart_rows["label"].value_counts())

wire_rows = df[df["text"].str.contains("21st Century Wire|21WIRE", na=False, case=False, regex=True)]
print("\nRows mentioning '21st Century Wire' / '21WIRE':", len(wire_rows))
print(wire_rows["label"].value_counts())p