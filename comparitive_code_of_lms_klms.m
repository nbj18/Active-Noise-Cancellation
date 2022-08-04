TD = 10;
a = 1;
np =.04;
N_tr = 500;
N_te = 100;


load MK30   
MK30 = MK30+np*randn(size(MK30));
MK30 = MK30 - mean(MK30);

train_set = MK30(1501:4500);

test_set = MK30(4601:4900);

X = zeros(TD,N_tr);
for k=1:N_tr
    X(:,k) = train_set(k:k+TD-1)';
end
T = train_set(TD+1:TD+N_tr);

X_te = zeros(TD,N_te);
for k=1:N_te
    X_te(:,k) = test_set(k:k+TD-1)';
end
T_te = test_set(TD+1:TD+N_te);


mse_te_l = zeros(N_tr,1);

%Linear LMS
lr_l = .2;%learning rate
w1 = zeros(1,TD);
e_l = zeros(N_tr,1);
for n=1:N_tr
    y = w1*X(:,n);
    e_l(n) = T(n) - y;
    w1 = w1 + lr_l*e_l(n)*X(:,n)';

    %testing MSE
    err_te = T_te'-(w1*X_te);
    mse_te_l(n) = mean(err_te.^2);
end



lr_k = .2;
%   experimental value 

%init
e_k = zeros(N_tr,1);
y = zeros(N_tr,1);
mse_te_k = zeros(N_tr,1);

% n=1 initialisation
e_k(1) = T(1);
y(1) = 0;
mse_te_k(1) = mean(T_te.^2);

for n=2:N_tr
    ii = 1:n-1;
    y(n) = lr_k*e_k(ii)'*(exp(-sum((X(:,n)*ones(1,n-1)-X(:,ii)).^2)))';
    e_k(n) = T(n) - y(n);
    
    %testing MSE
    y_te = zeros(N_te,1);
    for jj = 1:N_te
        y_te(jj) = lr_k*e_k(1:n)'*(exp(-sum((X_te(:,jj)*ones(1,n)-X(:,1:n)).^2)))';
    end
    err = T_te - y_te;
    mse_te_k(n) = mean(err.^2);
    
end

figure
plot(mse_te_l,'r-','LineWidth',2);
hold on
plot(mse_te_k,'k--','LineWidth',2);

set(gca, 'FontSize', 14);
set(gca, 'FontName', 'Arial');

legend('LMS', 'KLMS')
xlabel('iteration')
ylabel('MSE')