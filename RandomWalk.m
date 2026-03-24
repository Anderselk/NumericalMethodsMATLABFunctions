function [samples, means, vars] = Elk_Anders_hw2_prob1(M, N)
% Simulate Anisotropic 3D random walk x{n+1} = Rn*xn + un; x0 = 0

% u_n - unit vector in the xy plane
% R_n - 3D rotation by angle phi ~ U(0, 2*pi) around axis of u_n

% Inputs:
%   M - number of independent paths (Monte Carlo samples)
%   N - number of steps
% Outputs:
%   samples - Vector of |x_N| values, (Mx1), samples at final time N
%   means - Vector of E(|x_n|), ((N+1)x1), estimated from M samples, for n = 0 to N
%   vars - Vector of Var(|x_n|), ((N+1)x1), estimated from M samples, for n = 0 to N

    x = zeros(3, M); % holds current position of all M
    norms = zeros(N+1, M); % holds |x_n| for M at step n
    norms(1, :) = 0; % All norms start at 0

    for n = 1:N
        % draw random angles for all M 
        theta = 2 * pi * rand(1, M); % theta_n ~ U(0, 2*pi)
        phi = 2 * pi * rand(1, M); % phi_n ~ U(0, 2*pi)

        % Construct step vectors for all M
        u = [cos(theta); sin(theta); zeros(1, M)];

        % Apply rotation formula (calculation in written portion)
        udotx = sum(u .* x, 1);
        ucrossx = [u(2,:).*x(3,:); 
                  -u(1,:).*x(3,:); 
                   u(1,:).*x(2,:) - u(2,:).*x(1,:)];
        Rx = cos(phi) .* x + (1 - cos(phi)) .* udotx .* u + sin(phi) .* ucrossx;

        x = Rx + u; % Update Positions

        % Record Euclidean norms for all M
        norms(n+1, :) = sqrt(sum(x.^2, 1));
    end

    % Returns
    samples = norms(N+1, :)';
    means = mean(norms, 2);
    vars = var(norms, 0, 2);

end
