from models import train_classification, train_hurdle

def main():
    print("========== TRAINING CLASSIFICATION MODELS ==========")
    train_classification.main()
    
    print("\n========== TRAINING HURDLE MODELS ==========")
    train_hurdle.main()
    
    print("\nAll models trained and saved successfully!")

if __name__ == "__main__":
    main()
