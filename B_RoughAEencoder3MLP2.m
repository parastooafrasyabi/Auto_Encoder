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
%%
[n, m]=size(data);
input_num=m-1;
data_min=min(data);
data_max=max(data);
for i=1:n
    for j=1:m
        data(i,j)=(data(i,j)-data_min(1,j))/(data_max(1,j)-data_min(1,j));
    end
end


x=data(:,1:input_num);
%% Initialize Parameters
train_rate=0.7;
eta_e1=0.05;
eta_e2=0.03;
eta_e3=0.01;
eta_p=0.001;
epochs_ae=30;
max_epoch_p=100;

n0_neurons=input_num;
n1_neurons=64;
n2_neurons=32;
n3_neurons=16;

l1_neurons=8;
l2_neurons=1;

lowerb=-1;
upperb=1;

%% Initialize Autoencoder Weights
w_e1L = unifrnd(lowerb,upperb,[n1_neurons n0_neurons]);
w_e1U = unifrnd(lowerb,upperb,[n1_neurons n0_neurons]);
w_e2L = unifrnd(lowerb,upperb,[n2_neurons n1_neurons]);
w_e2U = unifrnd(lowerb,upperb,[n2_neurons n1_neurons]);
w_e3L = unifrnd(lowerb,upperb,[n3_neurons n2_neurons]);
w_e3U = unifrnd(lowerb,upperb,[n3_neurons n2_neurons]);

w_d1L=unifrnd(lowerb,upperb,[n0_neurons n1_neurons]);
w_d1U=unifrnd(lowerb,upperb,[n0_neurons n1_neurons]);
w_d1=0.5*(w_d1U+w_d1L);
w_d2L=unifrnd(lowerb,upperb,[n1_neurons n2_neurons]);
w_d2U=unifrnd(lowerb,upperb,[n1_neurons n2_neurons]);
w_d2=0.5*(w_d2U+w_d2L);
w_d3L=unifrnd(lowerb,upperb,[n2_neurons n3_neurons]);
w_d3U=unifrnd(lowerb,upperb,[n2_neurons n3_neurons]);
w_d3=0.5*(w_d3U+w_d3L);

%first Encoder
netL_e1=zeros(n1_neurons,1);
netU_e1=zeros(n1_neurons,1);

h1L=zeros(n1_neurons,1);
h1U=zeros(n1_neurons,1);

netL_d1=zeros(n0_neurons,1);
netU_d1=zeros(n0_neurons,1);

xL_hat=zeros(n0_neurons,1);
xU_hat=zeros(n0_neurons,1);

% second Encoder
netL_e2=zeros(n2_neurons,1);
netU_e2=zeros(n2_neurons,1);

h2L=zeros(n2_neurons,1);
h2U=zeros(n2_neurons,1);

netL_d2=zeros(n1_neurons,1);
netU_d2=zeros(n1_neurons,1);

h1L_hat=zeros(n1_neurons,1);
h1U_hat=zeros(n1_neurons,1);

%  third Encoder
netL_e3=zeros(n3_neurons,1);
netU_e3=zeros(n3_neurons,1);

h3L=zeros(n3_neurons,1);
h3U=zeros(n3_neurons,1);

netL_d3=zeros(n2_neurons,1);
netU_d3=zeros(n2_neurons,1);

h2L_hat=zeros(n2_neurons,1);
h2U_hat=zeros(n2_neurons,1);

%% noise
sigma= sqrt(0.3);
mu= 0;
noise=mu + sigma*(randn(n,m-1));
x=x+noise;
%% Encoder 1 Local Train
for i=epochs_ae
    for j=1:n
        % Feedforward
        
        % Encoder1
        netL_e1=w_e1L*x(j,:)';
        netU_e1=w_e1U*x(j,:)';

        h1L=logsig(netL_e1);
        h1U=logsig(netU_e1);
              h1=0.5*(h1L+h1U);

        
        % Decoder1
        netL_d1=w_d1L*h1;
        netU_d1=w_d1U*h1;
        
        xL_hat=logsig(netL_d1);
        xU_hat=logsig(netU_d1);
               x_hat=0.5*(xL_hat+xU_hat);
        
        % Error
        err=x(j,:)-x_hat';
        
             
        % Back Propagation
        
        f_derivative_dL=diag(xL_hat.*(1-xL_hat));
        f_derivative_dU=diag(xL_hat.*(1-xU_hat));
        f_derivative_d=0.5*(f_derivative_dU+f_derivative_dL);
        f_derivative_eL=diag(h1L.*(1-h1L));
        f_derivative_eU=diag(h1U.*(1-h1U));

        delta_w_d1L=(eta_e1 * h1 * err * f_derivative_dL)';
        delta_w_d1U=(eta_e1 * h1 * err * f_derivative_dU)';
        delta_w_e1L=(0.5*eta_e1 * x(j,:)' * err * f_derivative_d * w_d1 * f_derivative_eL)';
        delta_w_e1U=(0.5*eta_e1 * x(j,:)' * err * f_derivative_d * w_d1 * f_derivative_eU)';
          
        
        w_d1L = w_d1L + delta_w_d1L; 
        w_d1U = w_d1U + delta_w_d1U;
        w_d1=0.5*(w_d1U+w_d1L);
        w_e1L = w_e1L + delta_w_e1L;
        w_e1U = w_e1U + delta_w_e1U;


    end
end
%% Encoder 2 Local Train
for i=epochs_ae
    for j=1:n
        % Feedforward
        
        % Encoder1
        netL_e1=w_e1L*x(j,:)';
        netU_e1=w_e1U*x(j,:)';

         h1L = logsig(netL_e1);  % n1*1
         h1U = logsig(netU_e1);  % n1*1
         h1=0.5*(h1L+h1U);
         
        % Encoder2
        netL_e2=w_e2L*h1;
        netU_e2=w_e2U*h1;
         h2L=logsig(netL_e2);
         h2U=logsig(netU_e2);
         h2=0.5*(h2L+h2U);
         
        % Decoder2
        netL_d2=w_d2L*h2;
        netU_d2=w_d2U*h2;
        h1L_hat=logsig(netL_d2);
        h1U_hat=logsig(netU_d2);
        h1_hat=0.5*(h1U_hat+h1L_hat);
        % Error
        err=(h1-h1_hat)';
        
        % Back Propagation
        
        f_derivative_dL=diag(h1L_hat.*(1-h1L_hat));
        f_derivative_dU=diag(h1U_hat.*(1-h1U_hat));
        f_derivative_d=0.5*(f_derivative_dU+f_derivative_dL);
        f_derivative_eL=diag(h2L.*(1-h2L));
        f_derivative_eU=diag(h2U.*(1-h2U));

         delta_w_d2L = (eta_e2 * h2 * err * f_derivative_dL)';
         delta_w_d2U = (eta_e2 * h2 * err * f_derivative_dU)';
         delta_w_d2=0.5*(delta_w_d2U+delta_w_d2L);
         delta_w_e2L=(eta_e2 * h1 * err * f_derivative_d * w_d2 *0.5*f_derivative_eL)';
         delta_w_e2U=(eta_e2 * h1 * err * f_derivative_d * w_d2 *0.5* f_derivative_eU)';

         

        w_d2L = w_d2L + delta_w_d2L;
        w_d2U = w_d2U + delta_w_d2U;
        w_d2=0.5*(w_d2U+w_d2L);
        w_e2L = w_e2L + delta_w_e2L;
        w_e2U = w_e2U + delta_w_e2U;


    end
end
%% Encoder 3 Local Train
for i=epochs_ae
    for j=1:n
        % Feedforward
        
        % Encoder1
         netL_e1=w_e1L*x(j,:)';
         netU_e1=w_e1U*x(j,:)';

         h1L = logsig(netL_e1);  % n1*1
         h1U = logsig(netU_e1);  % n1*1
         h1=0.5*(h1L+h1U);

        % Encoder2
         netL_e2=w_e2L*h1;
         netU_e2=w_e2U*h1;
         h2L=logsig(netL_e2);
         h2U=logsig(netU_e2);
         h2=0.5*(h2L+h2U);

        % Encoder3
        netL_e3=w_e3L*h2;
        netU_e3=w_e3L*h2;
        
        h3L=logsig(netL_e3);
        h3U=logsig(netU_e3);
        h3=0.5*(h3L+h3U);

        % Decoder2
        netL_d3=w_d3L*h3;
        netU_d3=w_d3U*h3;
        net_d3=0.5*(netU_d3+netL_d3);
        h3L_hat=logsig(netL_d3);
        h3U_hat=logsig(netU_d3);
             h3_hat=0.5*(h3U_hat+h3L_hat);
        % Error
        err=(h2-h3_hat)';
        
        % Back Propagation
        
        f_derivative_dL=diag(h3L_hat.*(1-h3L_hat));
        f_derivative_dU=diag(h3U_hat.*(1-h3U_hat));
        f_derivative_d=0.5*(f_derivative_dU+f_derivative_dL);
        f_derivative_eL=diag(h3L.*(1-h3L));
        f_derivative_eU=diag(h3U.*(1-h3U));

        delta_w_d3L = (eta_e3 * h3 * err * f_derivative_dL)'; 
        delta_w_d3U = (eta_e3 * h3 * err * f_derivative_dU)';
         delta_w_d3=0.5*( delta_w_d3L+ delta_w_d3U);
        delta_w_e3L=(eta_e3 * h2 * err * f_derivative_d * w_d3 *0.5* f_derivative_eL)';
        delta_w_e3U=(eta_e3 * h2 * err * f_derivative_d * w_d3 *0.5* f_derivative_eU)';

        w_d3L = w_d3L + delta_w_d3L; 
        w_d3U = w_d3U + delta_w_d3U;
        w_d3=0.5*(w_d3U+w_d3L);
        w_e3L = w_e3L + delta_w_e3L;
        w_e3U = w_e3U + delta_w_e3U;

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

w2=unifrnd(lowerb,upperb,[l2_neurons l1_neurons]);
net2=zeros(l2_neurons,1);
o2=zeros(l2_neurons,1);

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
         netL_e1=w_e1L*input';
         netU_e1=w_e1U*input';

         h1L = logsig(netL_e1);  % n1*1
         h1U = logsig(netU_e1);  % n1*1
         h1=0.5*(h1L+h1U);

        % Encoder2
         netL_e2=w_e2L*h1;
         netU_e2=w_e2U*h1;
         h2L=logsig(netL_e2);
         h2U=logsig(netU_e2);
         h2=0.5*(h2L+h2U);

        % Encoder3
        netL_e3=w_e3L*h2;
        netU_e3=w_e3L*h2;
        h3L=logsig(netL_e3);
        h3U=logsig(netU_e3);
        h3=0.5*(h3L+h3U);

      % Layer 1
      net1=w1*h3;  % l1*1
      o1=logsig(net1);  % l1*1
      
      % Layer 2
      net2=w2*o1;  % l2*1
      o2=net2;  % l2*1
      
      % Predicted Output
      output_train(j,1)=o2;
      
      % Calc Error
      e=target-o2;  % 1*1
      error_train(j,1)=e;
      
      % Back Propagation
      f_driviate=diag(o1.*(1-o1)); % l1*l1 
      w1=w1-eta_p*e*-1*1*(w2*f_driviate)'*h3';  % l1*n0 = l1*n0 - 1*1 * 1*1 * (1*l1 * l1*l1)' * 1*n0
      w2=w2-eta_p*e*-1*1*o1';  % 1*l1 = 1*l1 - 1*1 * 1*1 * 1*l1
           
  end
  
  % Mean Square Train Error
  mse_train(i,1)=mse(error_train);
  
  % Test
  for j=1:num_of_test
      
      input=data_test(j,1:m-1);
      target=data_test(j,m);
      
      % Feed-Forward
 % Encoder1
         netL_e1=w_e1L*input';
         netU_e1=w_e1U*input';
         h1L = logsig(netL_e1);  % n1*1
         h1U = logsig(netU_e1);  % n1*1
         h1=0.5*(h1L+h1U);

        % Encoder2
         netL_e2=w_e2L*h1;
         netU_e2=w_e2U*h1;
         h2L=logsig(netL_e2);
         h2U=logsig(netU_e2);
         h2=0.5*(h2L+h2U);

        % Encoder3
        netL_e3=w_e3L*h2;
        netU_e3=w_e3L*h2;
        h3L=logsig(netL_e3);
        h3U=logsig(netU_e3);
        h3=0.5*(h3L+h3U);
      
      % Layer 1
      net1=w1*h3;  % l1*1
      o1=logsig(net1);  % l1*1
      
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

