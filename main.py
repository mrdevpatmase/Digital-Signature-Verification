from src.train import train
from src.evaluate import evaluate
from src.predict import predict

if __name__ == "__main__":
    print("1️⃣ Training Model...")
    model, history, classes = train()

    print("\n2️⃣ Evaluating Model...")
    evaluate()

    print("\n3️⃣ Testing with sample signature...")
    label, conf = predict(r"C:\Users\devpa\OneDrive\Desktop\Backup\Digital Signature Verification\src\sample.jpg", classes)
    print(f"Predicted: {label} ({conf*100:.2f}%)")
