#include <vector>
#include <random>
#include <map>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <queue>
#include <unistd.h>
#include <atomic>
#include "Eigen/Dense"

using std::cout;
using std::endl;

double calculateStandardError(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0f;
    }

    // Calculate the mean
    double mean = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();

    // Calculate the variance
    double variance = 0.0f;
    for (double value : values) {
        variance += (value - mean) * (value - mean);
    }
    variance /= values.size();

    // Return the standard deviation
    return std::sqrt(variance) / std::sqrt(values.size());
}

// Function to simulate a stock path
std::vector<std::vector<double>> simulate_stock_prices(double S0, double sigma, double r, double D, double dt, int N, int NSim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    std::vector<std::vector<double>> all_paths(NSim, std::vector<double>(N));
    for (int sim = 0; sim < NSim; ++sim) {
        all_paths[sim][0] = S0;
        for (int i = 1; i < N; ++i) {
            double drift = (r - D - 0.5 * sigma * sigma) * dt;
            double diffusion = sigma * sqrt(dt) * d(gen);
            all_paths[sim][i] = all_paths[sim][i - 1] * exp(drift + diffusion);
        }
    }
    return all_paths;
}

// process_path function
std::vector<double> american_option_pricing(std::vector<std::vector<double>>& SSit, double KP, double r, double dt, int N, int NSim, int numThreads) {
    // Matrix to hold option values for each simulation path
    std::vector<std::vector<double>> value_matrix(NSim, std::vector<double>(N, 0.0));
    // Matrix to hold payoffs for each simulation path
    std::vector<std::vector<double>> payoff_matrix(NSim, std::vector<double>(N, 0.0));

    // Initialize payoff matrix for each simulation path
    for (int sim = 0; sim < NSim; ++sim) {
        for (int t = 0; t < N; ++t) {
            payoff_matrix[sim][t] = std::max(KP - SSit[sim][t], 0.0);  // Calculating payoff for each timestep
        }
    }

    // Initialize the value_matrix with the payoffs at the final timestep
    for (int sim = 0; sim < NSim; ++sim) {
        value_matrix[sim].back() = payoff_matrix[sim].back();
    }

    // Working backwards through each time step
    for (int t = N - 2; t > 0; --t) {
        std::vector<double> X, Y;
        for (int sim = 0; sim < NSim; ++sim) {
            if (SSit[sim][t] < KP) { // In-the-money condition
                X.push_back(SSit[sim][t]);
                Y.push_back(value_matrix[sim][t + 1] * exp(-r * dt));
            }
        }

        if (X.empty()) continue;

        // Prepare the matrix for regression
        Eigen::MatrixXd matrixX(X.size(), 3);
        for (size_t i = 0; i < X.size(); ++i) {
            matrixX(i, 0) = X[i];        // X
            matrixX(i, 1) = X[i] * X[i]; // X^2
            matrixX(i, 2) = X[i] * X[i] * X[i]; // X^3
        }
        Eigen::VectorXd vectorY = Eigen::VectorXd::Map(Y.data(), Y.size());

        // Perform regression
        Eigen::VectorXd coefficients = matrixX.householderQr().solve(vectorY);

        // Iterate through each path
        #pragma omp parallel for num_threads(numThreads)
        for (int sim = 0; sim < NSim; ++sim) {
            if (SSit[sim][t] < KP) {
                double continuation_value = coefficients[0] * SSit[sim][t] + coefficients[1] * SSit[sim][t] * SSit[sim][t] + coefficients[2] * SSit[sim][t] * SSit[sim][t] * SSit[sim][t];
                double immediate_exercise_value = payoff_matrix[sim][t];

                // Decide whether to exercise
                if (immediate_exercise_value > continuation_value) {
                    value_matrix[sim][t] = immediate_exercise_value;
                    // Zero out future values since the option is exercised
                    std::fill(value_matrix[sim].begin() + t + 1, value_matrix[sim].end(), 0.0);
                }
            }
        }
    }

    // Calculating the option price
    std::vector<double> discounted_present_values;
    for (int sim = 0; sim < NSim; ++sim) {
//        for(auto v: value_matrix[sim]){
//            cout<<v<<", ";
//        }
//        cout<<endl;
        discounted_present_values.push_back(*std::max_element(value_matrix[sim].begin(), value_matrix[sim].end()));
    }

    return discounted_present_values; // Return the mean as the estimated option value
}

int main(int argc, char* argv[]){
    //Parameters
    double sigma = 0.2;  // Stock volatility
    double S0 = 80.0;  // Initial stock price
    double r = 0.04;  // Risk-free interest rate
    double D = 0.0;  // Dividend yield
    double T = 1;  // to maturity
    double KP = 100.0;  // Strike price
    double dt = 1.0 / 50;  // Time step size
    int N = int(T / dt);  // Number of time steps

    int NSim = 1000;
    int numThreads = 8;


    if (argc > 1) {
        numThreads = std::atoi(argv[1]);  // Override number of threads if provided
        if (argc > 2) {
            NSim = std::atoi(argv[2]);  // Override number of simulations if provided
        }
    }

    omp_set_num_threads(numThreads);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<double>> SSit = simulate_stock_prices(S0, sigma, r, D, dt, N, NSim);

    std::vector<double> result = american_option_pricing(SSit, KP, r, dt, N, NSim, numThreads);

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    double price = std::accumulate(result.begin(), result.end(), 0.0)/NSim;
    double stderror = calculateStandardError(result);

    cout << "Price: " << price <<", Standard Error: " << stderror << endl;
    cout << "Time taken: " << duration.count() << " milliseconds for " <<NSim<<" iterations with "<< omp_get_max_threads()<<" threads" << endl;

    return 0;
}



