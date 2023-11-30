# unit 13- Programming project
# sofia Torres COT4400


import time
import random
import matplotlib.pyplot as plt
import pandas as pd


# ALG1: Insertion Sort
def insertion_sort(A):
    start_time = time.time()
    for j in range(1, len(A)):
        key = A[j]
        i = j - 1
        while i >= 0 and A[i] > key:
            A[i + 1] = A[i]
            i = i - 1
        A[i + 1] = key
    end_time = time.time()
    return (end_time - start_time) * 1000  # Convert to milliseconds


# ALG2: Merge Sort
def merge_sort(A):
    def merge(A, left, mid, right):
        L = A[left : mid + 1]
        R = A[mid + 1 : right + 1]

        i = j = 0
        k = left

        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                A[k] = L[i]
                i += 1
            else:
                A[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            A[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            A[k] = R[j]
            j += 1
            k += 1

    def merge_sort_recursive(A, left, right):
        if left < right:
            mid = (left + right) // 2
            merge_sort_recursive(A, left, mid)
            merge_sort_recursive(A, mid + 1, right)
            merge(A, left, mid, right)

    start_time = time.time()
    merge_sort_recursive(A, 0, len(A) - 1)
    end_time = time.time()
    return (end_time - start_time) * 1000  # Convert to milliseconds


# ALG3: Randomized-Select
def randomized_partition(A, p, r):
    i = random.randint(p, r)
    A[i], A[r] = A[r], A[i]
    return partition(A, p, r)


def partition(A, p, r):
    x = A[r]
    i = p - 1

    for j in range(p, r):
        if A[j] <= x:
            i += 1
            A[i], A[j] = A[j], A[i]

    A[i + 1], A[r] = A[r], A[i + 1]
    return i + 1


def randomized_select(A, p, r, i):
    if p == r:
        return A[p]

    q = randomized_partition(A, p, r)
    k = q - p + 1

    if i == k:
        return A[q]
    elif i < k:
        return randomized_select(A, p, q - 1, i)
    else:
        return randomized_select(A, q + 1, r, i - k)


# Test Parameters
n_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
m = 5  # Number of iterations

# Lists to store results for each algorithm
empirical_runtimes_alg1 = []
empirical_runtimes_alg2 = []
empirical_runtimes_alg3 = []

predicted_runtimes_alg1 = []
predicted_runtimes_alg2 = []
predicted_runtimes_alg3 = []

# Test Script
for n in n_values:
    empirical_runtimes_alg1_n = []
    empirical_runtimes_alg2_n = []
    empirical_runtimes_alg3_n = []

    for _ in range(m):
        # Generate a random array for each iteration
        test_array = random.sample(range(1, n + 1), n)
        i = int(2 * n / 3)  # i = 2n/3

        # ALG1
        runtime_alg1 = insertion_sort(test_array.copy())
        empirical_runtimes_alg1_n.append(runtime_alg1)

        # ALG2
        runtime_alg2 = merge_sort(test_array.copy())
        empirical_runtimes_alg2_n.append(runtime_alg2)

        # ALG3
        try:
            runtime_alg3 = randomized_select(test_array.copy(), 0, n - 1, i)
            empirical_runtimes_alg3_n.append(runtime_alg3)
        except ValueError as e:
            print(f"Error in ALG3 for n = {n}: {e}")

    # Store average runtimes for each algorithm
    empirical_runtimes_alg1.append(sum(empirical_runtimes_alg1_n) / m)
    empirical_runtimes_alg2.append(sum(empirical_runtimes_alg2_n) / m)
    empirical_runtimes_alg3.append(sum(empirical_runtimes_alg3_n) / m)

    predicted_runtimes_alg1.append(empirical_runtimes_alg1[-1] / 2)
    predicted_runtimes_alg2.append(empirical_runtimes_alg2[-1] / 2)
    predicted_runtimes_alg3.append(empirical_runtimes_alg3[-1] / 2)

    # Print results for each input size
    print(f"\nInput Size n = {n}:")
    print(f"ALG1: Average RT - {empirical_runtimes_alg1[-1]:.3f} milliseconds")
    print(f"ALG2: Average RT - {empirical_runtimes_alg2[-1]:.3f} milliseconds")
    print(f"ALG3: Average RT - {empirical_runtimes_alg3[-1]:.3f} milliseconds")

# Graph 1 Empirical Runtimes of each algorithm
plt.plot(n_values, empirical_runtimes_alg1, label="ALG1")
plt.plot(n_values, empirical_runtimes_alg2, label="ALG2")
plt.plot(n_values, empirical_runtimes_alg3, label="ALG3")
plt.xlabel("Input Size (n)")
plt.ylabel("Empirical RT (milliseconds)")
plt.title("Empirical RT of Algorithms")
plt.legend()
plt.show()

# Graph 2 Empirical vs Predicted Runtimes for ALG1
plt.plot(n_values, empirical_runtimes_alg1, label="Empirical ALG1")
plt.plot(n_values, predicted_runtimes_alg1, label="Predicted ALG1")
plt.xlabel("Input Size (n)")
plt.ylabel("RT (milliseconds)")
plt.title("Empirical vs Predicted RT for ALG1")
plt.legend()
plt.show()

# Graph 3 Empirical vs Predicted Runtimes for ALG2
plt.plot(n_values, empirical_runtimes_alg2, label="Empirical ALG2")
plt.plot(n_values, predicted_runtimes_alg2, label="Predicted ALG2")
plt.xlabel("Input Size (n)")
plt.ylabel("RT (milliseconds)")
plt.title("Empirical vs Predicted RT for ALG2")
plt.legend()
plt.show()

# Graph 4 Empirical vs Predicted Runtimes for ALG3
plt.plot(n_values, empirical_runtimes_alg3, label="Empirical ALG3")
plt.plot(n_values, predicted_runtimes_alg3, label="Predicted ALG3")
plt.xlabel("Input Size (n)")
plt.ylabel("RT (milliseconds)")
plt.title("Empirical vs Predicted RT for ALG3")
plt.legend()
plt.show()


# Update tables for each algorithm
table_alg1 = pd.DataFrame(
    {
        "n": n_values,
        "TheoreticalRT": [
            n**2 for n in n_values
        ],  # Theoretical runtime for Insertion Sort
        "EmpiricalRT": empirical_runtimes_alg1,
        "Ratio": [
            empirical / (n**2)
            for empirical, n in zip(empirical_runtimes_alg1, n_values)
        ],
        "PredictedRT": predicted_runtimes_alg1,
    }
)

table_alg2 = pd.DataFrame(
    {
        "n": n_values,
        "TheoreticalRT": [
            n * (n_values[-1] // 1000) for n in n_values
        ],  # Theoretical runtime for Merge Sort
        "EmpiricalRT": empirical_runtimes_alg2,
        "Ratio": [
            empirical / (n * (n_values[-1] // 1000))
            for empirical, n in zip(empirical_runtimes_alg2, n_values)
        ],
        "PredictedRT": predicted_runtimes_alg2,
    }
)

table_alg3 = pd.DataFrame(
    {
        "n": n_values,
        "TheoreticalRT": [
            n * (n_values[-1] // 1000) for n in n_values
        ],  # Theoretical runtime for Randomized-Select
        "EmpiricalRT": empirical_runtimes_alg3,
        "Ratio": [
            empirical / (n * (n_values[-1] // 1000))
            for empirical, n in zip(empirical_runtimes_alg3, n_values)
        ],
        "PredictedRT": predicted_runtimes_alg3,
    }
)

# Display tables
print("\nTable for ALG1:")
print(table_alg1)

print("\nTable for ALG2:")
print(table_alg2)

print("\nTable for ALG3:")
print(table_alg3)



