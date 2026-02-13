from .train_kmeans import train_and_save

if __name__ == "__main__":
    rfm = train_and_save()
    print("Saved models. Segment counts:\n", rfm["Segment"].value_counts())