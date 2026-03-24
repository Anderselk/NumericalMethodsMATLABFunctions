function T = Elk_Anders_hw4_prob1(M, N, alpha, beta, gamma)
% Gillespie simulation of SIRS model

% Inputs:
%   M - # of simulation trials
%   N - fixed population size
%   alpha - infection rate
%   beta - recovery rate
%   gamma - losing immunity rate

% Results:
%   T - vector of extinction times

    T = zeros(M, 1);
    I0 = round(0.1 * N); %ICs
    S0 = N - I0;
    R0 = 0;

    
    for k = 1:M % M independent trials
        % Reset state for trial
        S = S0;  I = I0;  R = R0;  t = 0;

        while I > 0
            a1 = alpha * S * I; % R1: S + I => 2I 
            a2 = beta  * I;     % R2: I => R 
            a3 = gamma * R;     % R3: R => S 
            a0 = a1 + a2 + a3;  

            % Infinite runtime bugfix - if all rates 0, time goes inf, set
            if a0 == 0
                t = Inf;
                break;
            end

            % Waiting time from Exp(a0), inverse-CDF
            tau = -log(rand) / a0;
            t   = t + tau;

            % Choose reaction: uniform partitioned by summed rates
            u = rand * a0;
            if u < a1 % R1
                S = S - 1;
                I = I + 1;
            elseif u < a1 + a2 % R2
                I = I - 1;
                R = R + 1;
            else % R3
                R = R - 1;
                S = S + 1;
            end
        end
        T(k) = t;

    end
end
