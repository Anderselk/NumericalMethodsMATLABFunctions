function C = Elk_Anders_hw4_prob3(M, mu, sigma, T, K, S_0, dt)
%Estimates the Black-Scholes option price C via Monte Carlo:

% Inputs:
%   M - # of Monte Carlo samples
%   mu - mean interest rate (drift)
%   sigma - volatility
%   T - expiry time
%   K - strike price
%   S_0 - initial asset price
%   dt - step size
% Results:
%   C - vector of discounted payoff samples; 
%           (take mean(C) to estimate the option price).

    nSteps = round(T / dt); % # time steps correction
    S = S_0 * ones(M, 1); %Asset price paths

    % E-M loop
    for n = 1:nSteps
        Z = randn(M, 1); % Draw M standard normal

        % Euler Maruyama update
        S = S + mu .* S .* dt + sigma .* S .* sqrt(dt) .* Z;
    end

    % Discounted payoff for each sample
    C = exp(-mu * T) .* max(S - K, 0);

end 
