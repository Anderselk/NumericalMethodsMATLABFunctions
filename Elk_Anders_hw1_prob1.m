% Problem 1: Y=(U1/U2)
close all; clear; clc;


% PART B
m = 50;
Y = rand(m,1)./rand(m,1);
% uncomment below to display example from (b)
%display(Y) 


% PART D

% Define bin edges from (c) - see analytical soln
BinEdges = zeros(11,1);
for i = 0:10
    if i<=5
        BinEdges(i+1) = 0.2*i; % Account for MATLAB iteration at 1 (0_0)
    else
        BinEdges(i+1) = 1 / (2 * (1 - 0.1*i));
    end
end

% Adjust start and end for rounding error
BinEdges(1) = 0;   
BinEdges(end) = inf; 

% Print Bin Edges
fprintf('Bin edges y_i:\n');
for i = 1:11
    fprintf('y_%d = %.4f\n', i-1, BinEdges(i));
end
fprintf('\n');

% Analysis for different N values

N_Values = [10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8];
n_bins = 10;

% Code & Result Table
results = zeros(length(N_Values), 3);

fprintf('Part (d) Table:\n');
fprintf('%-12s %-20s %-20s\n', 'N', 'Std Dev', 'sqrt(N)*Std Dev');
fprintf('%s\n', repmat('-', 1, 60));

for ii = 1:length(N_Values)
    N = N_Values(ii);
    
    % Generate, count N samples of Y
    SamplesY = rand(N, 1) ./ rand(N, 1);
    count = histcounts(SamplesY, BinEdges);
    
    % Estimate probabilities
    Est_Prob = count / N;
    Theoretical_Prob = 0.1;
    
    % Calculate standard deviation of estimated probabilities
    StdDev = std(Est_Prob);
    SqrtNStd = sqrt(N) * StdDev;

    results(ii, 1) = N;
    results(ii, 2) = StdDev;
    results(ii, 3) = SqrtNStd;
    
    fprintf('%-12d %-20.6f %-20.6f\n', N, StdDev, SqrtNStd);
end

fprintf('\n');

% Theoretical limits:
% sqrt(N)*Std(p_hat) = sqrt(p(1-p)) = sqrt(0.09) = 0.3??
fprintf('Theoretical limit = sqrt(0.09) = %.4f\n', sqrt(0.09));
fprintf('\n');

% PART E

N_test = 100000; % Sample size
num_tests = 10000; % # of repetitions
alpha = 0.05; % Significance level

df = n_bins-1;
chi2_crit = chi2inv(1-alpha, df);

fprintf('Part (e): Chi-square GoF Test\n');
fprintf('Number of tests: %d\n', num_tests);
fprintf('Sample size/test: N = %d\n', N_test);
fprintf('Alpha = %.2f\n', alpha);
fprintf('Degrees of freedom: %d\n', df);
fprintf('Chi-square critical value: %.4f\n', chi2_crit);
fprintf('\n');

% Do chi-square test num_tests times
n_failures = 0;
chi2_stats = zeros(num_tests, 1);

for test = 1:num_tests
    SamplesY = rand(N_test, 1) ./ rand(N_test, 1);
    observed = histcounts(SamplesY, BinEdges);
    expected = N_test * 0.1 * ones(1, n_bins);
    
    chi2_stat = sum((observed - expected).^2 ./ expected);
    chi2_stats(test) = chi2_stat;
    
    if chi2_stat > chi2_crit
        n_failures = n_failures + 1;
    end
end

% Failure rate
failure_rate = n_failures / num_tests;

fprintf('Results:\n');
fprintf('Number of tests that failed: %d out of %d\n', n_failures, num_tests);
fprintf('Failure rate: %.4f (%.2f%%)\n', failure_rate, failure_rate*100);
fprintf('Expected failure rate (alpha): %.4f (%.2f%%)\n', alpha, alpha*100);
fprintf('\n');

% Stat summary
fprintf('Chi-square statistics summary:\n');
fprintf('Mean: %.4f (Expected: %.4f)\n', mean(chi2_stats), df);
fprintf('Std:  %.4f (Expected: %.4f)\n', std(chi2_stats), sqrt(2*df));
fprintf('\n');