function [votecounts] = Elk_Anders_hw3_prob2(M)
% Metropolis-Hastings MCMC sampler over voting district configurations
% for the state of 'Wissquaresin'.

% Input:
%   M - number of MCMC steps (length of output chain)

% Output:
%   votecounts - each entry is the number of districts won by party '1' 
%                                            in that configuration

    %VOTING PREFERENCES from the given file
    V = [0.89 0.83 0.64 0.58 0.61 0.58 0.57 0.75 0.81 0.75 0.84 0.89;
         0.71 0.71 0.70 0.74 0.69 0.49 0.34 0.57 0.73 0.58 0.63 0.70;
         0.30 0.39 0.63 0.61 0.35 0.15 0.10 0.23 0.59 0.60 0.54 0.36;
         0.13 0.21 0.44 0.32 0.12 0.04 0.03 0.11 0.45 0.66 0.50 0.19;
         0.18 0.30 0.46 0.27 0.09 0.04 0.02 0.05 0.22 0.50 0.30 0.14;
         0.55 0.75 0.78 0.58 0.25 0.13 0.05 0.06 0.20 0.40 0.20 0.20;
         0.78 0.92 0.92 0.84 0.61 0.40 0.15 0.15 0.37 0.46 0.27 0.33;
         0.85 0.93 0.90 0.84 0.70 0.57 0.35 0.31 0.46 0.46 0.46 0.55;
         0.83 0.82 0.63 0.47 0.33 0.41 0.48 0.39 0.42 0.46 0.61 0.70;
         0.72 0.58 0.24 0.12 0.12 0.33 0.67 0.68 0.71 0.70 0.71 0.71;
         0.71 0.48 0.19 0.09 0.09 0.26 0.70 0.81 0.84 0.84 0.86 0.82;
         0.88 0.77 0.49 0.28 0.22 0.32 0.61 0.76 0.82 0.84 0.90 0.91];

    N = 12; % grid dimension
    Np = N*N; % total # of precincts
    Nd = 8; % # of districts
    v = V(:); % flatten preferences to vector

    % ADJACENCY MATRIX 
    % Two nodes are adjacent iff they share a face.
    diag1 = repmat([ones(1, N-1) 0], 1, N);
    A = diag(diag1(1:end-1), 1) + diag(diag1(1:end-1), -1) + ...
        diag(ones(1, Np-N), N) + diag(ones(1, Np-N), -N);
    A = sparse(A); % Store sparse

    districts = zeros(Np, 1); 
    for d = 1:Nd
        districts((d-1)*18 + 1 : d*18) = d;
    end

    % Helper Func 1: Compute vote outcome for current district assignment
    function wins = compute_votes(dist, preferences)
        wins = 0;
        for dd = 1:Nd
            ii = (dist == dd);
            n_precincts = sum(ii);
            vote_share = sum(preferences(ii));
            if vote_share > 0.5 * n_precincts
                wins = wins + 1;
            end
        end
    end

    % Helper Func 2: Find all boundary edges
    function B = get_boundary_edges(dist)
        % Use upper triu of A to not double count
        [rows, cols] = find(triu(A));
        mask = dist(rows) ~= dist(cols);
        B = [rows(mask), cols(mask)];
    end

    % Helper Func 3: Check continuity of single district after removing a precinct.
    function ok = is_contiguous(dist, d, removed_node)
        % Nodes remaining in d after removing removed_node
        members = find(dist == d);
        members = members(members ~= removed_node);
        if numel(members) == 0
            ok = false; return;
        end
        if isscalar(members)
            ok = true; return;
        end
        % Subset matrix to remaining
        Asub = A(members, members);
        G = graph(Asub);
        cc = conncomp(G);
        ok = (max(cc) == 1);
    end
    
    %Main Func Loop
    votecounts = zeros(M, 1);
    B = get_boundary_edges(districts);

    for step = 1:M

        nB = size(B, 1);  % # of edges

        % Propose a move
        edge_idx = randi(nB);
        edge = B(edge_idx, :);

        % Pick 1 of 2 endpoint nodes to flip
        if rand < 0.5
            target = edge(1);
            new_district = districts(edge(2));
        else
            target = edge(2);
            new_district = districts(edge(1));
        end

        old_district = districts(target);

        % If same district, skip
        if old_district == new_district
            votecounts(step) = compute_votes(districts, v);
            continue;
        end

        % Compute boundary-edge neighbors of target in x
        neighbors_of_target = find(A(:, target));
        NT_x = sum(districts(neighbors_of_target) ~= old_district);

        %Test size constraints
        old_size = sum(districts == old_district);
        new_size = sum(districts == new_district);

        if (old_size - 1) < 17 || (new_size + 1) > 19
            votecounts(step) = compute_votes(districts, v);
            continue;
        end

        %Check Continuous of donor district
        if ~is_contiguous(districts, old_district, target)
            votecounts(step) = compute_votes(districts, v);
            continue;
        end

        %  Compute proposed configuration y
        districts_y = districts;
        districts_y(target) = new_district;

        % Compute boundary-edge neighbors of target in y
        NT_y = sum(districts_y(neighbors_of_target) ~= new_district);

        %Compute B(y)
        B_y = get_boundary_edges(districts_y);
        nB_y = size(B_y, 1);

        % MH acceptance ratio
        log_alpha = log(NT_y) + log(nB) - log(NT_x) - log(nB_y);

        % Accept or reject
        if log(rand) < log_alpha
            districts = districts_y;
            B = B_y;
        end

        % Record vote count by step
        votecounts(step) = compute_votes(districts, v);
    end
end