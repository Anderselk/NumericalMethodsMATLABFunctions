function [samples] = Elk_Anders_hw3_prob1(nsamples, nchains, nskip, lambda)
% Metropolis-Hastings sampler for a Poisson(lambda) distribution.

% Inputs:
%   nsamples - # of samples to collect per chain (rows)
%   nchains - # of independent chains (columns)
%   nskip - burn-in: skip first nskip steps, then thin by keeping
%                       every nskip-th step to reduce autocorrelation
%   lambda - Poisson rate parameter

% Output:
%   samples - (nsamples x nchains) matrix of samples

    total_steps = nskip + nsamples * nskip;

    % Initialize all chains at round(lambda), approx mode of distribution
    state = round(lambda) * ones(1, nchains);
    samples = zeros(nsamples, nchains);
    sample_count = 0;

    % Main Loop - Iterate, After burn-in, save sample every nskip steps.
    for step = 1:total_steps
        % Draw uniform(0,1) for each chain to decide direction
        u_direction = rand(1, nchains);

        % Where state > 0, step right with prob 1/2, left with prob 1/2
        % Where state = 0, always step right (no left)
        proposed = state + ones(1, nchains); % default step right
        step_left = (u_direction >= 0.5) & (state > 0); % valid left step
        proposed(step_left) = state(step_left) - 1;

        log_pi_ratio = zeros(1, nchains); %log[pi(proposed)/pi(current)]

        % Chains stepping right (proposed = state + 1)
        stepping_right = ~step_left;
        log_pi_ratio(stepping_right) = log(lambda) - log(proposed(stepping_right));

        % Chains stepping left (proposed = state - 1, state >= 1)
        log_pi_ratio(step_left) = -log(lambda) + log(state(step_left));

        log_proposal_ratio = zeros(1, nchains);

        % Case A: stepping right from k=0
        from_zero = (state == 0); %right to 1
        log_proposal_ratio(from_zero) = -log(2);

        % Case B: stepping left from k=1 to k=0
        to_zero = step_left & (state == 1);
        log_proposal_ratio(to_zero) = log(2);

        % Total log acceptance
        log_alpha = log_pi_ratio + log_proposal_ratio;

        % Accept if log(U)<logalpha
        log_u = log(rand(1, nchains));
        accept = (log_u < log_alpha);

        % Update states(accepted chains move to proposed, others stay)
        state(accept) = proposed(accept);

        % Skip first nskip steps, after burn-in, every nskip-th step.
        if step > nskip
            step_after_burnin = step - nskip; % steps since burn-in ended
            if mod(step_after_burnin, nskip) == 0
                sample_count = sample_count + 1;
                samples(sample_count, :) = state;
            end
        end
    end 
end 