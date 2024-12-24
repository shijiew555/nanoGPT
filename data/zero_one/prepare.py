import time
from generate_dataset import generate_dataset
from load_dataset import load_dataset


start_time = time.time()
# generate csv file containing dataset
generate_dataset()
end_time = time.time()
print(f"Dataset generation time: {end_time - start_time} seconds")

start_time = time.time()
# load dataset into .bin file
load_dataset()
end_time = time.time()
print(f"Dataset load time: {end_time - start_time} seconds")
