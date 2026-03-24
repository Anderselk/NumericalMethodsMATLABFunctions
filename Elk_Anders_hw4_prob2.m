function X = Elk_Anders_hw4_prob2(M, T, sigma, omega, dt)
% Simulate samples of X(T) for Brownian, driven by correlated noise.
% Inputs:
%   M - # of independent samples
%   T - final simulation time
%   sigma - noise amplitude of the correlated noise driving process
%   omega - mean-reversion rate G - correlation time of 1/omega
%   dt - step size
%
% Results:
%   X - vector of positions x(T)

    % Fixed Params
    alpha = 16;
    I0alpha = besseli(0, alpha); % Bessel function first kind I0(alpha)

    x0 = acos(log(I0alpha) / alpha); % ICs

    % Compute # of steps
    nSteps = round(T / dt); % integer-round to handle floats

    % Initialise
    X = x0 * ones(M, 1); % position
    G = zeros(M, 1); % correlated noise

    % E-M loop over time steps
    for n = 1:nSteps
        % Evaluate force f(X) at current positions
        fX = exp(alpha .* cos(X)) / I0alpha - 1;

        % Draw independent standard normal increments for update
        Z = randn(M, 1); %Zn ~ N(0,1) for each

        % Update correlated noise process G
        G = G - omega .* G .* dt + sigma .* sqrt(dt) .* Z;

        % Update position X
        X = X + fX .* dt + G .* dt;
    end
end