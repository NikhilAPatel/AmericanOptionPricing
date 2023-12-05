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


// Global variables and data structures
struct Timestep {
    int index;
    int lastUpdated; // This could be a timestamp or a simple counter

    Timestep(int idx, double time) : index(idx), lastUpdated(time) {}
};

// Comparator for the priority queue
struct CompareTimestep {
    bool operator()(const Timestep& a, const Timestep& b) {
        return a.lastUpdated > b.lastUpdated; // Min heap (smaller lastUpdated has higher priority)
    }
};


std::atomic<bool> adjustLoad(false);
std::atomic<int> numSimulators(16);

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
std::vector<double> simulate_stock_path(double S0, double sigma, double r, double D, double dt, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    std::vector<double> path(N);
    path[0] = S0;
    for (int i = 1; i < N; ++i) {
        double drift = (r - D - 0.5 * sigma * sigma) * dt;
        double diffusion = sigma * sqrt(dt) * d(gen);
        path[i] = path[i - 1] * exp(drift + diffusion);
    }
    return path;
}

// process_path function
void process_path(double S0, double sigma, double r, double dt, double D, int N, double KP, int P, int NSimulators, std::vector<double>& results, std::vector<std::vector<std::pair<int, double>>>& dataQueue, std::vector<std::vector<double>>& coefficientsMap) {
    int simsCompleted = 0;

    //Initialize stuff needed for random number generation
    std::default_random_engine generator;
    generator.seed(omp_get_thread_num());

    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::normal_distribution<double> normaldistribution(0.0,1.0);

    auto SSit = simulate_stock_path(S0, sigma, r, D, dt, N);

    std::vector<double> payoff(N);
    std::transform(SSit.begin(), SSit.end(), payoff.begin(), [KP](double s) { return std::max(KP - s, 0.0); });
    std::vector<double> value_matrix(N, 0.0);
    value_matrix.back() = payoff.back();

    for (int t = N - 2; t > 0; --t) { //TODO not sure if this should be t>0 or t>=0 (when >=, we get a lot of prices to be 9 exactly in the first index, which suggests something fishy)
        bool in_the_money = SSit[t] < KP;
        if (in_the_money) {
            // With probability 1/P, send data to the global queue
            if (distribution(generator) < 1.0 / NSimulators) {
                double discounted_value = value_matrix[t + 1] * exp(-r);
                #pragma omp critical
                {
                    dataQueue[t].emplace_back(SSit[t], discounted_value);
                }
            }


            std::vector<double> coefficients;

            //Important: removing this critical section allows for linear scaling with p
//                    #pragma omp critical
//                    {
//                        coefficients = get_regression_coefficients(t);
//                        coefficients = {normaldistribution(generators[threadId]), normaldistribution(generators[threadId]),normaldistribution(generators[threadId]),normaldistribution(generators[threadId]),};
            coefficients = coefficientsMap[t];
//                    }



            // Calculate continuation value
            double continuation_value = 0.0;
            for (size_t i = 0; i < coefficients.size(); ++i) {
                continuation_value += coefficients[i] * std::pow(SSit[t], i);
            }

            double immediate_exercise_value = payoff[t];

            // Decide whether to exercise
            if (immediate_exercise_value > continuation_value) {
                value_matrix[t] = immediate_exercise_value;
                std::fill(value_matrix.begin() + t + 1, value_matrix.end(),
                          0.0); // Zero out future values
            }
        }
    }
    double price = *std::max_element(value_matrix.begin(), value_matrix.end());

    #pragma omp critical
    {
        simsCompleted++;
        results.emplace_back(price);
    }
}

void regression_worker(std::vector<std::vector<std::pair<int, double>>>& dataQueue, std::vector<std::vector<double>>& coefficientsMap, std::priority_queue<Timestep, std::vector<Timestep>, CompareTimestep>& pq, int regression_timestep){
    int timestep;
    std::vector<std::pair<int, double>> training_data;

    //Get timestep from priority queue and
    //Grabbing data from shared queue
    #pragma omp critical
    {
        if(!pq.empty()){
            timestep = pq.top().index;
            pq.pop();
        }
        training_data = dataQueue[timestep];
    }

    // Setting up the matrices and vectors
    Eigen::MatrixXd X(training_data.size(), 3);
    Eigen::VectorXd Y(training_data.size());

    for (size_t i = 0; i < training_data.size(); ++i) {
        double x = training_data[i].first;
        X(i, 0) = x;      // X
        X(i, 1) = x * x;  // X^2
        X(i, 2) = x * x * x;  // X^3
        Y(i) = training_data[i].second;
    }

    // Performing the regression
    Eigen::VectorXd coefficients = X.householderQr().solve(Y);
    std::vector<double> stdCoefficients(coefficients.data(), coefficients.data() + coefficients.size());

    //Putting the coefficients in the map
    //Putting the timestep back in the priority queue
    #pragma omp critical
    {
        coefficientsMap[timestep] = stdCoefficients;
        pq.push(Timestep(timestep, regression_timestep));
    }
}

void adjustThreadRoles(int simsCompleted, int regsCompleted){
    double ratio = simsCompleted / (double)std::max(1, regsCompleted);

    if (ratio > 3) {
        // Too many simulations, increase regressors
        numSimulators = std::max(1, numSimulators - 1); // Decrease simulator count
    } else {
        // Too few simulations, decrease regressors
        numSimulators = std::min(numSimulators+1, omp_get_max_threads()); // Increase simulator count
    }

}

std::vector<double> runSimulation(double S0, double sigma, double r, double dt, double D, int N, double KP, int NSim){
    omp_set_nested(1); //Allows each parallel section below to spawn their own threads
    omp_set_num_threads(32);

    std::vector<double> results;
    std::vector<std::vector<std::pair<int, double>>> dataQueue(N);
    std::vector<std::vector<double>> coefficientsMap(N);

    std::priority_queue<Timestep, std::vector<Timestep>, CompareTimestep> pq;
    for (int i = 0; i < N; ++i) {
        pq.push(Timestep(i, 0));
    }

    int regression_timestep=0;

    int simulationsCompleted = 0;

    #pragma omp parallel
    {
        while (simulationsCompleted<NSim){
            if(omp_get_thread_num() < numSimulators){
                process_path(S0, sigma, r, dt, D, N, KP, numSimulators, numSimulators, results, dataQueue, coefficientsMap);

                #pragma omp critical
                {
                    simulationsCompleted++;
                }

            }else{
                regression_worker(dataQueue, coefficientsMap, pq, regression_timestep);

                #pragma omp critical
                {
                    regression_timestep++;
                }
            }

            #pragma omp master
            {
                adjustThreadRoles(simulationsCompleted, regression_timestep);
            }
        }
    }
    return results;
}

int main(int argc, char* argv[]){
    //Parameters
    double sigma = 1.0;  // Stock volatility
    double S0 = 80.0;  // Initial stock price
    double r = 0.04;  // Risk-free interest rate
    double D = 0.0;  // Dividend yield
    double T = 1;  // to maturity
    double KP = 160.0;  // Strike price
    double dt = 1.0 / 50;  // Time step size
    int N = int(T / dt);  // Number of time steps
    int NSim = 100000;  // Default number of simulation paths (can be overridden through command line)

    // Read NSim if provided
    if (argc > 1) {
        NSim = std::atoi(argv[1]);
    }

    cout<<NSim<<endl;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> result = runSimulation(S0, sigma, r, dt, D, N, KP, NSim);

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    double price = std::accumulate(result.begin(), result.end(), 0.0)/NSim;
    double stderror = calculateStandardError(result);

    cout << "Price: " << price <<", Standard Error: " << stderror << endl;
    cout << "Time taken: " << duration.count() << " milliseconds for " <<NSim<<" iterations with " << omp_get_max_threads() <<" threads "<< endl;

    return 0;
}



