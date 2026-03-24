function [pdfvals, coefvars] = Elk_Anders_hw2_prob2(M, nbins)
% Multiple importance sampling estimate of the PDF of |x_10| 
% for the anisotropic 3D random walk.

% Standard walk has theta_n ~ U(0,2*pi). Bias theta_n toward
% the projection of x_n onto the (x,y) plane. (See the written portion.)
% Multiple importance sampling with K values of alpha is combined using
% balance heuristic, cover range [0,10] of |x_10|.

% Inputs:
%   M - number of samples per biasing distribution
%   nbins - number of bins over [0, 10]
% Outputs:
%   pdfvals - vector of estimated PDF values at bin centers, (nbinsx1) 
%   coefvars - vector of coefficients of variation per bin, (nbinsx1) 

    N = 10;
    xmax = 10;
    dx = xmax/nbins;

    % Set Values of alpha to try to sample uniformly across [0,10]
    alphavec = [1, 2, 4, 8, 9]; 
    K = length(alphavec);

    % final_norms(k,j) - |x_10| for sample j under generating dist k
    %log_pstar_all(k,l,j) - sum over steps of log p*[alpha_l](delta_n^(k,j))
    finalnorms = zeros(K, M);
    log_pstar_all = zeros(K, K, M); 
    log_pval = -N * log(2*pi); 

    for k = 1:K
        alpha_k = alphavec(k);
        x = zeros(3, M);
        % Accumulate log p*[alpha_l] over steps for each sample and l
        log_pstarbatch = zeros(K, M);

        for n = 0:(N-1)
            psi = atan2(x(2,:), x(1,:)); % Angle of projection of x_n
            % Since origin undefined, theta uniformly, weight this step is p/p* = 1
            atorigin = (x(1,:) == 0) & (x(2,:) == 0);

            % Draw Xn ~ U(0,1), form biased angular deviation
            Xn = rand(1, M);
            Yn = 2*Xn - 1; 
            delta = pi*sign(Yn) .* abs(Yn).^alpha_k;

            % Biased step angle, fix origin due to deviation
            theta = psi + delta;
            theta(atorigin) = 2*pi*rand(1, sum(atorigin));

            % Accumulate log p*[alpha_l](delta) for each proposal l.
            for l = 1:K
                alph = alphavec(l);
                lp = -log(2*alph*pi) + (1/alph - 1)*log(max(abs(delta), eps)/pi);
                lp(atorigin) = log(1/(2*pi));
                log_pstarbatch(l,:) = log_pstarbatch(l,:) + lp;
            end

            % Draw phin ~ U(0,2*pi) for rotation
            phi = 2*pi*rand(1, M);
            cos_phi = cos(phi);
            sin_phi = sin(phi);

            u = [cos(theta); sin(theta); zeros(1, M)]; % Step unit vectors

            udotx = sum(u .* x, 1); %Stuff from problem 1
            ucrossx = [ u(2,:).*x(3,:);  
                       -u(1,:).*x(3,:); 
                        u(1,:).*x(2,:) - u(2,:).*x(1,:)];
            Rx = cos_phi .*x + (1-cos_phi).* udotx .*u + sin_phi.* ucrossx;

            x = Rx + u; %Update Positions
        end

        % Store norms, accumulated log p* by batch
        finalnorms(k,:)    = sqrt(sum(x.^2, 1));
        log_pstar_all(k,:,:) = log_pstarbatch; 
    end 

    % For sample j from generating distribution k:
    log_w_bal = zeros(K, M);
    for k = 1:K
        %squeeze funct bugfix, (extra dim of 1??)
        log_pstar_kj = squeeze(log_pstar_all(k,:,:)); 

        % log-sum-exp over proposal distributions
        X_max = max(log_pstar_kj, [], 1);
        lse = log(sum(exp(log_pstar_kj - X_max), 1)) + X_max;
        log_w_bal(k,:) = log_pval + log(K) - lse;
    end

    w_bal = exp(log_w_bal);

    all_norms = finalnorms(:); % Flatten all results to vectors
    all_w = w_bal(:); 
    total = K * M;

    % Assign each sample to bins, add upper bound into last bin
    bin_idx = min(floor(all_norms / dx) + 1, nbins);

    wsum = accumarray(bin_idx, all_w, [nbins, 1]); % PDF estimator
    pdfvals = wsum / (total * dx); 

    % Coefficient of variation by variance of indiv. contributions.
    % Each sample j contributes cj = all-w(j)*I[bin=i] to its bin.
    w2sum = accumarray(bin_idx, all_w.^2, [nbins, 1]);
    Ec = wsum  / total;
    Ec2 = w2sum / total;
    var_c = max(Ec2 - Ec.^2, 0) * total/(total-1);
    std_pdf = sqrt(var_c) / (sqrt(total) * dx);
    coefvars = std_pdf ./ (pdfvals + eps); 
end