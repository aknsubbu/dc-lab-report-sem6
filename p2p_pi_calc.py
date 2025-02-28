from mpi4py import MPI
import math
import time

def f(x):
    """Function to integrate: 4/(1 + x^2)"""
    return 4.0 / (1.0 + x * x)

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Get process rank
    size = comm.Get_size()  # Get number of processes

    # Parameters
    n = 1000000  # Number of intervals

    # Start timing on rank 0
    start_time = time.time() if rank == 0 else None

    # Calculate width of each interval
    h = 1.0 / n

    # Each process computes its portion of the sum
    local_sum = 0.0
    for i in range(rank, n, size):
        x = h * (i + 0.5)  # Midpoint of the interval
        local_sum += f(x)
    local_sum *= h  # Multiply by width to get area

    # Reduce all local sums to the global sum on rank 0
    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    # Output result from rank 0
    if rank == 0:
        pi = global_sum
        end_time = time.time()
        print(f"Calculated value of pi: {pi:.16f}")
        print(f"Error: {abs(pi - math.pi):.16f}")
        print(f"Time taken: {end_time - start_time:.6f} seconds")

if __name__ == "__main__":
    main()
