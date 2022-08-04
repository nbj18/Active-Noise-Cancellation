clear all,
close all
clc
%time delay (embedding) length
TD = 5;
D = 2;
h = .1;%kernel size

u = randn(1,2500)>0;% Generate binary data
u = 2*u-1;

% Nonlinear channel
z = u+0.5*[0,u(1:end-1)];
% Channel noise
ns = 0.4*randn(1,length(u));
% Ouput of the nonlinear channel
y = z - 0.9*z.^2 + ns;

%data size
N_tr = 1000;
N_te = 50;

%data embedding
X = zeros(TD,N_tr);
for k=1:N_tr
    X(:,k) = y(k:k+TD-1)';
end
% Test data
X_te = zeros(TD,N_te);
for k=1:N_te
    X_te(:,k) = y(k+N_tr:k+TD-1+N_tr)';
end

% Desired signal
T = zeros(N_tr,1);
for ii=1:N_tr
    T(ii) = u(D+ii);
end

T_te = zeros(N_te,1);
for ii=1:N_te
    T_te(ii) = u(D+ii+N_tr);
end

mse_te = zeros(1,N_tr);
mse_te_k = zeros(1,N_tr);

lr_k = .2;
%init
e_k = zeros(N_tr,1);
y = zeros(N_tr,1);
y_te = zeros(N_te,1);
% n=1 init
e_k(1) = T(1);
y(1) = 0;
mse_te_k(1) = mean(T_te.^2);
% start
for n=2:N_tr
    %training
    ii = 1:n-1;
    y(n) = lr_k*e_k(ii)'*(exp(-sum((X(:,n)*ones(1,n-1)-X(:,ii)).^2)*h))';
    
    e_k(n) = T(n) - y(n);
    
    %testing
    y_te = zeros(N_te,1);
    for jj = 1:N_te
        ii = 1:n;
        y_te(jj) = lr_k*e_k(ii)'*(exp(-sum((X_te(:,jj)*ones(1,n)-X(:,ii)).^2)*h))';
    end
    err = T_te - y_te;
    mse_te_k(n) = mean(err.^2);
end

figure
plot(mse_te_k,'b-','LineWidth',2)
hold on
%
set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');

legend('KLMS')
xlabel('iteration')
ylabel('testing MSE')