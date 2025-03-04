from mpi4py import MPI
import math
import time
import sys

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
    
    # Each process prints its initialization info
    print(f"Process {rank}/{size-1} initialized on {MPI.Get_processor_name()}")
    sys.stdout.flush()  # Force output to be displayed immediately
    
    # Synchronize before timing
    comm.Barrier()
    
    # Start timing on all ranks to measure load balancing
    local_start = time.time()

    # Calculate width of each interval
    h = 1.0 / n
    
    # Calculate workload distribution stats
    intervals_per_process = n // size
    remainder = n % size
    my_intervals = intervals_per_process + (1 if rank < remainder else 0)
    start_interval = rank * intervals_per_process + min(rank, remainder)
    end_interval = start_interval + my_intervals
    
    if rank == 0:
        print(f"Total intervals: {n}")
        print(f"Dividing work among {size} processes")
        print(f"Base intervals per process: {intervals_per_process}")
        if remainder > 0:
            print(f"First {remainder} processes handle 1 extra interval")
    
    # Calculate the local sum with workload tracking
    local_sum = 0.0
    count = 0
    
    for i in range(start_interval, end_interval):
        x = h * (i + 0.5)  # Midpoint of the interval
        local_sum += f(x)
        count += 1
    
    local_sum *= h  # Multiply by width to get area
    
    # Calculate local processing time
    local_time = time.time() - local_start
    
    # Gather timing and workload info from all processes
    all_times = comm.gather(local_time, root=0)
    all_counts = comm.gather(count, root=0)
    
    # Reduce all local sums to the global sum on rank 0
    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    # Output result from rank 0
    if rank == 0:
        pi = global_sum
        end_time = time.time()
        total_time = end_time - local_start
        
        print("\n--- RESULTS ---")
        print(f"Calculated value of pi: {pi:.16f}")
        print(f"Error: {abs(pi - math.pi):.16f}")
        print(f"Total time taken: {total_time:.6f} seconds")
        
        # Performance metrics
        print("\n--- PERFORMANCE METRICS ---")
        min_time = min(all_times)
        max_time = max(all_times)
        avg_time = sum(all_times) / len(all_times)
        load_imbalance = max_time / min_time if min_time > 0 else float('inf')
        
        print(f"Fastest process time: {min_time:.6f} seconds")
        print(f"Slowest process time: {max_time:.6f} seconds")
        print(f"Average process time: {avg_time:.6f} seconds")
        print(f"Load imbalance factor: {load_imbalance:.2f}x")
        print(f"Parallel efficiency: {avg_time/total_time*100:.1f}%")
        
        # Workload distribution
        print("\n--- WORKLOAD DISTRIBUTION ---")
        for i in range(size):
            print(f"Process {i}: computed {all_counts[i]} intervals ({all_counts[i]/n*100:.1f}% of total)")

if __name__ == "__main__":
    main()