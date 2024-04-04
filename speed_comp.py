import cv2
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from numba import njit

from lw1 import (
    cv2_threshold,
    adaptive_threshold_python,
)


def timeit(func):
    def inner(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        end = time()
        time_diff = end - start
        return time_diff, res
    return inner


def main():
    
    image = cv2.imread("images/opened-book-gray.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # parameters for adaptive threshold
    block_sizes = range(3, 255, 2)
    C = 23
    max_val = 255
    
    e1_time = []
    e2_time = []
    e3_time = []

    for bs in tqdm(block_sizes):

        t1, _ = timeit(cv2_threshold)(image, max_val, bs, C)
        t2, _ = timeit(adaptive_threshold_python)(image, max_val, bs, C)
        t3, _ = timeit(njit(adaptive_threshold_python))(image, max_val, bs, C)
                 
        
        e1_time.append(t1)
        e2_time.append(t2)
        e3_time.append(t3)
        
    X = block_sizes
    
    plt.figure(figsize=(15,5))
    plt.plot(X, e1_time, color='red', label='OpenCV')
    plt.plot(X, e2_time, color='blue', label='Python+Numpy')
    plt.plot(X, e3_time, color='green', label='Python+Numpy+Numba')

    # Adding labels and legend
    plt.xlabel('Block size')
    plt.ylabel('Time, (s)')
    plt.title('Comparison of time')
    plt.legend()
    # Displaying the plot
    plt.show()

if __name__ == "__main__":
    main()