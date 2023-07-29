%Autoencoder_Local
clc;
close all;
clear;
%% Read and Making Data
x=xlsread('NDMT_Melbourne');

%%
 N = size(x,1);
 D=20;
  L=1;
[Data,Target]=Make_Data(x,N,D,L);
data=[Data,Target];
%  data=xlsread('Mackey-Glass');
%% Normalize Dataset
 m = size(data,2);
 n= size(data,1);
 input_num = m-1;
data_min = min(data);
data_max = max(data);
for i = 1:n
    for j = 1:m
        data(i, j) = (data(i, j) - data_min(1,j)) / (data_max(1,j) - data_min(1,j));
    end
end

 x = data(:, 1:input_num);
%% Initialize Parameters
train_rate=0.7;
eta_e1 = 0.05;
eta_e2 = 0.03;
eta_e3 = 0.01;
eta_p = 0.001;
epochs_ae = 30;
max_epoch_p = 1000;

n0_neurons = input_num;
n1_neurons =32;
n2_neurons =16;
n3_neurons=8;
l1_neurons=4;
l2_neurons=1;

lowerb=-1;
upperb=1;

%% Initialize Autoencoder weigths
w_e1 = unifrnd(lowerb,upperb,[n1_neurons n0_neurons]);
g_e1=unifrnd(0,1,[n1_neurons,1]);
w_e2 = unifrnd(lowerb,upperb,[n2_neurons n1_neurons]);
g_e2=unifrnd(0,1,[n2_neurons,1]);
w_e3 = unifrnd(lowerb,upperb,[n3_neurons n2_neurons]);
g_e3=unifrnd(0,1,[n3_neurons,1]);
w_d1 = unifrnd(lowerb,upperb,[n0_neurons n1_neurons]);
g_d1=unifrnd(0,1,[n0_neurons,1]);
w_d2 = unifrnd(lowerb,upperb,[n1_neurons n2_neurons]);
g_d2=unifrnd(0,1,[n1_neurons,1]);
w_d3 = unifrnd(lowerb,upperb,[n2_neurons n3_neurons]);
g_d3=unifrnd(0,1,[n2_neurons,1]);
net_e1=zeros(n1_neurons,1);
h1=zeros(n1_neurons,1);
net_d1=zeros(n0_neurons,1);
x_hat=zeros(n0_neurons,1);

net_e2=zeros(n2_neurons,1);
h2=zeros(n2_neurons,1);
net_d2=zeros(n1_neurons,1);
h1_hat=zeros(n1_neurons,1);

net_e3=zeros(n3_neurons,1);
h3=zeros(n3_neurons,1);
net_d3=zeros(n2_neurons,1);
h2_hat=zeros(n2_neurons,1);
%% Encoder 1 Local Train
for i=1:epochs_ae
    for j=1:n
        % Feed-Forward

        % Encoder1
        net_e1 = w_e1* x(j,:)';  % n1*1
%         h1 = logsig(net_e1);  % n1*1
h1 = logsig(net_e1.*g_e1);
        % Decoder1
        net_d1 = w_d1 * h1;  % n0*1
%         x_hat = logsig(net_d1);  % n0*1
x_hat = logsig(net_d1.*g_d1);
        % Error
        err = x(j,:) - x_hat';  % 1*n0

        % Back propagation
        f_driviate_d = diag(g_d1.*(x_hat.*(1-x_hat)));  % n0*n0
        df_g_d1=diag(net_d1.*(x_hat.*(1-x_hat)));
        f_driviate_e = diag(g_e1.*(h1.*(1-h1)));  % n1*n1
        df_g_e1=diag(net_e1.*(h1.*(1-h1)));
        delta_w_d1 = (eta_e1 * h1 * err * f_driviate_d)';  % n0*n1 = (1*1 * n1*1 * 1*n0 * n0*n0)'
        delta_w_e1 = (eta_e1 * x(j,:)' * err * f_driviate_d * w_d1 * f_driviate_e)';  % n1*n0 = (1*1 * n0*1 * 1*n0 * n0*n0 * n0*n1 * n1*n1)'
        w_d1 = w_d1 + delta_w_d1;  % n0*n1
        w_e1 = w_e1 + delta_w_e1;  % n1*n0
              g_d1=g_d1-(eta_e1*err*-1*df_g_d1)';
              g_e1=g_e1-(eta_e1*err*-1*f_driviate_d* w_d1*df_g_e1)';

    end
end

%% Encoder 2 Local Train
for i=1:epochs_ae
    for j=1:n
        % Feed-Forward

        % Encoder1
        net_e1 = w_e1* x(j,:)';  % n1*1
%         h1 = logsig(net_e1);  % n1*1
        h1 = logsig(net_e1.*g_e1);

        % Encoder2
        net_e2 = w_e2* h1;  % n2*1
%         h2 = logsig(net_e2);  % n2*1
        h2 = logsig(net_e2.*g_e2);

        % Decoder2
        net_d2 = w_d2 * h2;  % n1*1
%         h1_hat = logsig(net_d2);  % n1*1
        h1_hat = logsig(net_d2.*g_d2);
        % Error
        err = (h1 - h1_hat)';  % 1*n1

        % Back propagation
        f_driviate_d = diag((g_d2.*h1_hat.*(1-h1_hat)));  % n1*n1
        f_driviate_e = diag((g_e2.*h2.*(1-h2)));  % n2*n2
        df_g_d2=diag(net_d2.*(h1_hat.*(1-h1_hat)));
        df_g_e2=diag(net_e2.*(h2.*(1-h2)));
        delta_w_d2 = (eta_e2 * h2 * err * f_driviate_d)';  % n1*n2 = (1*1 * n2*1 * 1*n1 * n1*n1)'
        delta_w_e2 = (eta_e2 * h1 * err * f_driviate_d * w_d2 * f_driviate_e)';  % n2*n1 = (1*1 * n1*1 * 1*n1 * n1*n1 * n1*n2 * n2*n2)'
        w_d2 = w_d2 + delta_w_d2;  % n1*n2
        w_e2 = w_e2 + delta_w_e2;  % n2*n1
              g_d2=g_d2-(eta_e1*err*-1*df_g_d2)';
              g_e2=g_e2-(eta_e1*err*-1*f_driviate_d* w_d2*df_g_e2)';
        
    end
end
%% Encoder 3 Local Train
for i=1:epochs_ae
    for j=1:n
        % Feed-Forward

        % Encoder1
        net_e1 = w_e1* x(j,:)';  % n1*1
%         h1 = logsig(net_e1);  % n1*1
                h1 = logsig(net_e1.*g_e1);

        % Encoder2
        net_e2 = w_e2* h1;  % n2*1
%         h2 = logsig(net_e2);  % n2*1
                h2 = logsig(net_e2.*g_e2);

        % Encoder3
        net_e3 = w_e3* h2;  % n3*1
%         h3 = logsig(net_e3);  % n3*1
                        h3 = logsig(net_e3.*g_e3);

        % Decoder3
        net_d3 = w_d3 * h3;  % n2*1
%         h2_hat = logsig(net_d3);  % n2*1
        h2_hat = logsig(net_d3.*g_d3);

        % Error
        err = (h2 - h2_hat)';  % 1*n2
% 
        % Back propagation
        f_driviate_d = diag(g_d3.*(h2_hat.*(1-h2_hat)));  % n2*n2
        f_driviate_e = diag(g_e3.*(h3.*(1-h3)));  % n3*n3
        df_g_d3=diag(net_d3.*(h2_hat.*(1-h2_hat)));
        df_g_e3=diag(net_e3.*(h3.*(1-h3)));
        
        delta_w_d3 = (eta_e3 * h3 * err * f_driviate_d)';  % n2*n3 = (1*1 * n3*1 * 1*n2 * n2*n2)'
        delta_w_e3 = (eta_e3 * h2 * err * f_driviate_d * w_d3 * f_driviate_e)';  % n3*n2 = (1*1 * n2*1 * 1*n2 * n2*n2 * n2*n3 * n3*n3)'
        w_d3 = w_d3 + delta_w_d3;  % n2*n3
        w_e3 = w_e3 + delta_w_e3;  % n3*n2
              g_d3=g_d3-(eta_e1*err*-1*df_g_d3)';
              g_e3=g_e3-(eta_e1*err*-1*f_driviate_d* w_d3*df_g_e3)';
        
    end
end
%% Initialize Train and Test Data
num_of_train=round(train_rate*n);
num_of_test=n-num_of_train;
        
data_train=data(1:num_of_train,:);
data_test=data(num_of_train+1:n,:);


%% Initialize Perceptron weigths
w1=unifrnd(lowerb,upperb,[l1_neurons n3_neurons]);
net1=zeros(l1_neurons,1);
o1=zeros(l1_neurons,1);
g1=unifrnd(0,1,[l1_neurons,1]);

w2=unifrnd(lowerb,upperb,[l2_neurons l1_neurons]);
net2=zeros(l2_neurons,1);
o2=zeros(l2_neurons,1);
% g2=unifrnd(0,1,[l2_neurons,1]);

error_train=zeros(num_of_train,1);
error_test=zeros(num_of_test,1);

output_train=zeros(num_of_train,1);
output_test=zeros(num_of_test,1);

mse_train=zeros(max_epoch_p,1);
mse_test=zeros(max_epoch_p,1);

%% 2 Layer Perceptron
% Train
for i=1:max_epoch_p
  for j=1:num_of_train
      
      input=data_train(j,1:m-1);
      target=data_train(j,m);

      % Feed-Forward

      % Encoder1
      net_e1 = w_e1* input';  % n1*1
%       h1 = logsig(net_e1);  % n1*1
                h1 = logsig(net_e1.*g_e1);

      % Encoder2
      net_e2 = w_e2* h1;  % n2*1
%       h2 = logsig(net_e2);  % n2*1
                h2 = logsig(net_e2.*g_e2);

      % Encoder3
      net_e3 = w_e3* h2;  % n3*1
%       h3 = logsig(net_e3);  % n3*1
                      h3 = logsig(net_e3.*g_e3);

      % Layer 1
      net1=w1*h3;  % l1*1
%       o1=logsig(net1);  % l1*1
       o1 = logsig(net1.*g1);
      % Layer 2
      net2=w2*o1;  % l2*1
      o2=net2;  % l2*1
      
      % Predicted Output
      output_train(j,1)=o2;
      
      % Calc Error
      e=target-o2;  % 1*1
      error_train(j,1)=e;
      
      % Back Propagation
      f_driviate=diag(g1.*(o1.*(1-o1))); % l1*l1 
      df_g1=diag(net1.*(o1.*(1-o1)));
      w1=w1-eta_p*e*-1*1*(w2*f_driviate)'*h3';  % l1*n0 = l1*n0 - 1*1 * 1*1 * (1*l1 * l1*l1)' * 1*n0
      w2=w2-eta_p*e*-1*1*o1';  % 1*l1 = 1*l1 - 1*1 * 1*1 * 1*l1
      g1=g1-eta_p*e*-1*1*(w2*df_g1)';

  end
  
  % Mean Square Train Error
  mse_train(i,1)=mse(error_train);
  
  % Test
  for j=1:num_of_test
      
      input=data_test(j,1:m-1);
      target=data_test(j,m);
      
      % Feed-Forward

      % Encoder1
      net_e1 = w_e1* input';  % n1*1
%       h1 = logsig(net_e1);  % n1*1
                h1 = logsig(net_e1.*g_e1);

      % Encoder2
      net_e2 = w_e2* h1;  % n2*1
%       h2 = logsig(net_e2);  % n2*1
                      h2 = logsig(net_e2.*g_e2);

       % Encoder3
      net_e3 = w_e3* h2;  % n3*1
%       h3 = logsig(net_e3);  % n3*1
                      h3 = logsig(net_e3.*g_e3);

      % Layer 1
      net1=w1*h3;  % l1*1
%       o1=logsig(net1);  % l1*1
                      o1 = logsig(net1.*g1);

      % Layer 2
      net2=w2*o1;  % l2*1
      o2=net2;  % l2*1
      
      % Predicted Output
      output_test(j,1)=o2;
      
      % Calc Error
      e=target-o2;  % 1*1
      error_test(j,1)=e;     
      
  end 
  
  % Mean Square Test Error
  mse_test(i,1)=mse(error_test);
  
  %% Find Regression
  [m_train ,b_train]=polyfit(data_train(:,m),output_train(:,1),1);
  [y_fit_train,~] = polyval(m_train,data_train(:,m),b_train);
  [m_test ,b_test]=polyfit(data_test(:,m),output_test(:,1),1);
  [y_fit_test,~] = polyval(m_test,data_test(:,m),b_test);
  
  %% Plot Results
  figure(1);
  subplot(2,3,1),plot(data_train(:,m),'-r');
  hold on;
  subplot(2,3,1),plot(output_train,'-b');
  title('Output Train')
  hold off;
  
  subplot(2,3,2),semilogy(mse_train(1:i,1),'-r');
  title('MSE Train')
  hold off;
  
  subplot(2,3,3),plot(data_train(:,m),output_train(:,1),'b*');hold on;
  plot(data_train(:,m),y_fit_train,'r-');
  title('Regression Train')
  hold off;
  
  subplot(2,3,4),plot(data_test(:,m),'-r');
  hold on;
  subplot(2,3,4),plot(output_test,'-b');
  title('Output Test')
  hold off;
  
  subplot(2,3,5),plot(mse_test(1:i,1),'-r');
  title('MSE Test')
  hold off;
  
  subplot(2,3,6),plot(data_test(:,m),output_test(:,1),'b*');hold on;
  plot(data_test(:,m),y_fit_test,'r-');
  title('Regression Test')
  hold off;
  
  pause(0.001);
end

fprintf('mse train = %1.16g, mse test = %1.16g \n', mse_train(max_epoch_p,1), mse_test(max_epoch_p,1))
