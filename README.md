# AmericanOptionPricing
Parallelizing Monte Carlo LSM for Path-Dependent Options

- ClassicalSequential: The classical sequential Monte Carlo LSM algorithm
- EasyParallelization: Runs multiple versions of the classical simulator simultaneously with smaller batches. This will ensure a linear speedup with P, but will cause a drop in accuracy as each regression trained will not have access to the full batch of data.
- HybridParallelization: Smartly applies parallelization to the classical approach by parallelizing the update step after regressions are trained.
- AutoManager: Attempts to remedy the shortcomings of EasyParallelization by splitting up threads between being simulators or regressors. A manager dynamically allocates threads between simulating and regressing, attempting to strike a ratio between paths simulated and regressions completed in order to optimize both simulation time and accuracy.
