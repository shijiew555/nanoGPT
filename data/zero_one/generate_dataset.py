import os
import numpy as np

# Function to generate a sequence with probability 2/3 for 1 and 1/3 for 0
def generate_sequence(length, p_1=2/3):
    return np.random.choice([0, 1], size=length, p=[1 - p_1, p_1])

def generate_dataset():
    # Parameters for train.csv
    sequence_length = 1000  # Length of each sequence
    num_sequences = 10000  # Total number of sequences
    output_file = "train.csv"  # Output file name

    # remove existing train.csv file
    if os.path.exists("train.csv"):
        os.remove("train.csv")

    # Write sequences to file
    with open(output_file, "w") as f:
        f.write("text\n")
        for _ in range(num_sequences):
            sequence = generate_sequence(sequence_length)
            line = " ".join(map(str, sequence))  # Convert sequence to comma-separated string
            f.write(line + "\n")

    print(f"Generated {num_sequences} sequences and saved to {output_file}")

