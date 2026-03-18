data = load("/Users/arnavoruganty/Downloads/NMSE_convergence.mat");

figure;
hold on;

% -----------------------------
% LA-MCTS
% -----------------------------
plot(data.call_marks, data.best_fx, ...
    '-', 'LineWidth', 3);

% -----------------------------
% SA
% -----------------------------
plot(data.sa_evals, data.sa_history, ...
    '-', 'LineWidth', 3);

% -----------------------------
% PSO
% -----------------------------
plot(data.pso_evals, data.pso_history, ...
    '-', 'LineWidth', 3);

% -----------------------------
% Formatting (IEEE style)
% -----------------------------
set(gca, 'YScale', 'log');   % log NMSE
grid on;

xlabel('Function Evaluations', 'FontWeight', 'bold');
ylabel('NMSE', 'FontWeight', 'bold');

legend({'LA-MCTS (Proposed)', 'SA', 'PSO'}, ...
    'Location', 'best', ...
    'FontWeight', 'bold');

xlim([0 data.max_evals]);

set(gca, 'LineWidth', 1.5);  % axis thickness
set(gca, 'FontSize', 12);

box on;