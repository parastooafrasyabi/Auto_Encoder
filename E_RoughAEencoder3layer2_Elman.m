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
eta_e1=0.006;
eta_e2=0.004;
eta_e3=0.002;
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
w_e1 = unifrnd(lowerb,upperb,[n1_neurons n0_neurons]);
w_e2 = unifrnd(lowerb,upperb,[n2_neurons n1_neurons]);
w_e3 = unifrnd(lowerb,upperb,[n3_neurons n2_neurons]);

w_r1=unifrnd(lowerb,upperb,[n1_neurons n1_neurons]);
w_r2=unifrnd(lowerb,upperb,[n2_neurons n2_neurons]);
w_r3=unifrnd(lowerb,upperb,[n3_neurons n3_neurons]);

w_d1=unifrnd(lowerb,upperb,[n0_neurons n1_neurons]);
w_d2=unifrnd(lowerb,upperb,[n1_neurons n2_neurons]);
w_d3=unifrnd(lowerb,upperb,[n2_neurons n3_neurons]);

%first Encoder
net_e1=zeros(n1_neurons,1);

h1=zeros(n1_neurons,1);

h1_1takhir=zeros(n1_neurons,1);
h1_2takhir=zeros(n1_neurons,1);
net_d1=zeros(n0_neurons,1);
x_hat=zeros(n0_neurons,1);
x_hat_1takhir=zeros(n0_neurons,1);
x_hat_2takhir=zeros(n0_neurons,1);

f_derivative_d1_1takhir=zeros(n0_neurons,n0_neurons);
f_derivative_e1_1takhir =zeros(n1_neurons,n1_neurons);

wd1_1takhir=zeros(n0_neurons,n1_neurons);

% second Encoder
net_e2=zeros(n2_neurons,1);

h2=zeros(n2_neurons,1);
h2_1takhir=zeros(n2_neurons,1);
h2_2takhir=zeros(n2_neurons,1);

net_d2=zeros(n1_neurons,1);

h1_hat=zeros(n1_neurons,1);
h1_hat_1takhir=zeros(n1_neurons,1);
h1_hat_2takhir=zeros(n1_neurons,1);

f_derivative_d2_1takhir=zeros(n1_neurons,n1_neurons);
f_derivative_e2_1takhir =zeros(n2_neurons,n2_neurons);

wd2_1takhir=zeros(n1_neurons,n2_neurons);

%  third Encoder
net_e3=zeros(n3_neurons,1);

h3=zeros(n3_neurons,1);
h3_1takhir=zeros(n3_neurons,1);
h3_2takhir=zeros(n3_neurons,1);

net_d3=zeros(n2_neurons,1);

h2_hat=zeros(n2_neurons,1);
h2_hat_1takhir=zeros(n2_neurons,1);
h2_hat_2takhir=zeros(n2_neurons,1);

f_derivative_d3_1takhir=zeros(n2_neurons,n2_neurons);
f_derivative_e3_1takhir =zeros(n3_neurons,n3_neurons);

wd3_1takhir=zeros(n2_neurons,n3_neurons);

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
        net_e1=w_e1*x(j,:)'+w_r1*h1_1takhir;
        h1=logsig(net_e1);
                       
        % Decoder1
        net_d1=w_d1*h1;
        x_hat=logsig(net_d1);
        
        % Error
        err=x(j,:)-x_hat';
        
        % Back Propagation
        
        f_derivative_d=diag(x_hat.*(1-x_hat));
        f_derivative_e=diag(h1.*(1-h1));

        delta_w_d1=(eta_e1 * h1 * err * f_derivative_d)';
        delta_w_e1=(0.5*eta_e1 * x(j,:)' * err * f_derivative_d * w_d1 * f_derivative_e)';
        delta_w_r =eta_e1*(h1_1takhir+w_r1*0.5*f_derivative_e1_1takhir*h1_2takhir)*(err*f_derivative_d*w_d1*0.5*f_derivative_e);  
        
        w_r1 = w_r1 + delta_w_r;
        w_d1 = w_d1 + delta_w_d1; 
        w_e1 = w_e1 + delta_w_e1;
             
%%      Memory
        h1_2takhir= h1_1takhir;
        h1_1takhir= h1;
        
        x_hat_2takhir=x_hat_1takhir;
        x_hat_1takhir=x_hat;

        f_derivative_e1_1takhir=f_derivative_e;
        f_derivative_d1_1takhir= f_derivative_d;

        wd1_1takhir=w_d1;
    end
end
%% Encoder 2 Local Train
for i=epochs_ae
    for j=1:n
        % Feedforward
        
        % Encoder1
        net_e1=w_e1*x(j,:)'+w_r1*h1_1takhir;
        h1 = logsig(net_e1);  % n1*1
        
        % Encoder2
        net_e2=w_e2*h1+w_r2*h2_1takhir;
        h2=logsig(net_e2);

        % Decoder2
        net_d2=w_d2*h2;
        h1_hat=logsig(net_d2);
        
        % Error
        err=(h1-h1_hat)';
        
        % Back Propagation
        f_derivative_d=diag(h1_hat.*(1-h1_hat));
        f_derivative_e=diag(h2.*(1-h2));

         delta_w_d2 = (eta_e2 * h2 * err * f_derivative_d)';  
         delta_w_e2=(eta_e2 * h1 * err * f_derivative_d * w_d2 *0.5*f_derivative_e)';
         delta_w_r =eta_e2*(h2_1takhir+w_r2*0.5*f_derivative_e2_1takhir*h2_2takhir)*(err*f_derivative_d*w_d2*0.5*f_derivative_e); 
              
        w_r2 = w_r2 + delta_w_r; 
        w_d2 = w_d2 + delta_w_d2; 
        w_e2 = w_e2 + delta_w_e2;
               
%%      Memory
        h2_2takhir= h2_1takhir;
        h2_1takhir= h2;
        h1_1takhir= h1;

        h1_hat_2takhir=h1_hat_1takhir;
        h1_hat_1takhir=h1_hat;

        f_derivative_e2_1takhir=f_derivative_e;
        f_derivative_d2_1takhir= f_derivative_d;

        wd2_1takhir=w_d2;
    end
end
%% Encoder 3 Local Train
for i=epochs_ae
    for j=1:n
        % Feedforward
        
        % Encoder1
         net_e1=w_e1*x(j,:)'+w_r1*h1_1takhir;
         h1 = logsig(net_e1);  % n1*1
        
        % Encoder2
         net_e2=w_e2*h1+w_r2*h2_1takhir;
         h2=logsig(net_e2);

        % Encoder3
        net_e3=w_e3*h2+w_r3*h3_1takhir;
        h3=logsig(net_e3);

        % Decoder2
        net_d3=w_d3*h3;
        h3_hat=logsig(net_d3);
        
        % Error
        err=(h2-h2_hat)';
        
        % Back Propagation
        f_derivative_d=diag(h2_hat.*(1-h2_hat));
        f_derivative_e=diag(h3.*(1-h3));

        delta_w_d3 = (eta_e3 * h3 * err * f_derivative_d)';  
        delta_w_e3=(eta_e3 * h2 * err * f_derivative_d * w_d3 *0.5* f_derivative_e)';
        delta_w_r =eta_e3*(h3_1takhir+w_r3*0.5*f_derivative_e3_1takhir*h3_2takhir)*(err*f_derivative_d*w_d3*0.5*f_derivative_e);  
              
        w_r3 = w_r3 + delta_w_r;       
        w_d3 = w_d3 + delta_w_d3; 
        w_e3 = w_e3 + delta_w_e3;  
%%      Memory
        h3_2takhir= h3_1takhir;
        h3_1takhir= h3;
        h2_1takhir= h2;
        h1_1takhir= h1;

        h2_hat_2takhir=h2_hat_1takhir;
        h2_hat_1takhir=h2_hat;

        f_derivative_e3_1takhir=f_derivative_e;
        f_derivative_d3_1takhir= f_derivative_d;
       
        wd3_1takhir=w_d3;
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
         net_e1=w_e1*input'+w_r1*h1_1takhir;
         h1 = logsig(net_e1);  % n1*1
        
        % Encoder2
         net_e2=w_e2*h1+w_r2*h2_1takhir;
         h2=logsig(net_e2);
        

        % Encoder3
        net_e3=w_e3*h2+w_r3*h3_1takhir;
        h3=logsig(net_e3);

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
 
      %% Memory
        h3_1takhir= h3;
        h2_1takhir= h2;
        h1_1takhir= h1;
  end
  
  % Mean Square Train Error
  mse_train(i,1)=mse(error_train);
  
  % Test
  for j=1:num_of_test
      
      input=data_test(j,1:m-1);
      target=data_test(j,m);
      
      % Feed-Forward
 % Encoder1
         net_e1=w_e1*input'+w_r1*h1_1takhir;
         h1 = logsig(net_e1);  % n1*1
        
        % Encoder2
         net_e2=w_e2*h1+w_r2*h2_1takhir;
         h2=logsig(net_e2);
       
        % Encoder3
        net_e3=w_e3*h2+w_r3*h3_1takhir;
        h3=logsig(net_e3);
      
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
 %% Memory
   h3_1takhir= h3;
   h2_1takhir= h2;
   h1_1takhir= h1;
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

