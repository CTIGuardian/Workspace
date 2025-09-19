from datasets import load_dataset, DatasetDict

# Path to your CSV file
csv_path = "APTQA Dataset.csv"

# Load CSV into a Hugging Face Dataset
dataset = load_dataset("csv", data_files=csv_path)

# By default, this loads into a split called "train"
train_dataset = dataset["train"]

# Define the split ratios
validation_ratio = 0.2
test_ratio = 0.1

# Calculate sizes
num_examples = len(train_dataset)
num_validation = int(num_examples * validation_ratio)
num_test = int(num_examples * test_ratio)

# First split into train + (validation+test)
train_test_split = train_dataset.train_test_split(test_size=num_test + num_validation, seed=42)

# Then split (validation+test) into validation and test
validation_test_split = train_test_split["test"].train_test_split(
    test_size=num_test / (num_test + num_validation),
    seed=42
)

# Put everything together into a DatasetDict
final_dataset = DatasetDict({
    "train": train_test_split["train"],
    "validation": validation_test_split["train"],
    "test": validation_test_split["test"],
})

# Save to disk (optional)
final_dataset.save_to_disk("APTQA_Dataset")

print(final_dataset)
