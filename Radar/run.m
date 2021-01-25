load data/Amazon.mat;
start_time=cputime;
[n,~] = size(X);
X = normalizeFea(X, 0);
niters = 20;

alpha = 0.1;
beta = 0.01;
gamma = 0.1;
At = A';
Anew = max(A,At);
L = computelaplacian(Anew, 'undirected');
R = radar(X, A, L, alpha, beta, gamma, niters);
score= sum(R.*R,2);
[~,idx] = sort(score, 'descend');

gnd_data = zeros(n,2);
gnd_data(:,1) = gnd;
gnd_data(:,2) = score;
end_time=cputime;
op_time=end_time-start_time
[tp,fp] = roc([gnd_data(:,1),gnd_data(:,2)]);
plot(fp,tp);
auc_value = auc(gnd_data);