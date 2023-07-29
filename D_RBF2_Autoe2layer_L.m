clc;
close all;
clear;
%% Read and Normalize Dataset
x=xlsread('NDMT_Melbourne');
 N = size(x,1);
 D=15;
  L=4;
[Data,Target]=Make_Data(x,N,D,L);
data=[Data,Target];
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

eta_w1=0.001;
eta_m1=0.001;
eta_s1=0.001;
eta_w2=0.008;
eta_m2=0.008;
eta_s2=0.008;
eta_w3=0.008;
eta_m3=0.001;
eta_s3=0.001;


epochs_ae=30;
max_epoch_p=1000;

n0=input_num;
n1=8;
n2=6;
n3=4;

l1_neurons=2;
l2_neurons=1;

lowerb=-0.1;
upperb=0.1;

%% Initialize Autoencoder Weights
mean1 = rand([n1 n0]);
sigma1 = rand([n1 1]);
w_d1=rand([n0 n1]);

mean2 = rand([n2 n1]);
sigma2 = rand( [n2 1]);
w_d2=rand([n1 n2]);
%% Encoder 1 Local Train
for i=epochs_ae
    for j=1:n
        % Feedforward
        
        % Encoder1
        net_e1=sum((x(j,:) - mean1).^2, 2);
        h1=exp(-0.5 * net_e1./(sigma1.^2));
        
        % Decoder1
        net_d1=w_d1*h1;
        x_hat=(net_d1);
        
        % Error
        err=x(j,:)-x_hat';
        
        % Back Propagation


        delta_mean1 = ((eta_m1*err*-1*w_d1*(x(j,:) - mean1))'*(h1./(sigma1.^2))')'; 
        delta_sigma1=eta_s1*err*-1*w_d1*sum((x(j,:) - mean1).^2, 2).*(1./(sigma1.^3)).*h1;  
        mean1=mean1-delta_mean1;
        sigma1=sigma1-delta_sigma1;
        w_d1=w_d1-eta_w1*err'*-1*1*h1';


        
    end

end
%% Encoder 2 Local Train
for i=epochs_ae
    for j=1:n
        % Feedforward

        % Encoder1
        net_e1=sum((x(j,:) - mean1).^2, 2);
        h1=exp(-0.5 * net_e1./(sigma1.^2));
        
        % Encoder2
        net_e2=sum((h1' - mean2).^2, 2);
        h2=exp(-0.5 * net_e2./(sigma2.^2));
        
        % Decoder1
        net_d2=w_d2*h2;
        h1_hat=(net_d2);
        
        % Error
        err=(h1-h1_hat)';
        
%         Back Propagation
          delta_mean2 = ((eta_m2*err*-1*w_d2*(h1' - mean2))'*(h2./(sigma2.^2))')';  
          delta_sigma2=eta_s2*err*-1*w_d2*sum((h1' - mean2).^2, 2).*(1./(sigma2.^3)).*h2; 

          mean2=mean2-delta_mean2;
          sigma2=sigma2-delta_sigma2;
          w_d2=w_d2-eta_w2*err'*-1*1*h2';
          
        
    end

end
%% Initialize train and test data
num_of_train=round(train_rate*n);
num_of_test=n-num_of_train;

data_train=data(1:num_of_train,:);
data_test=data(num_of_train+1:n,:);

%% Initialize perceptrn weights
mean3=unifrnd(lowerb,upperb,[l1_neurons n2]);
sigma3=unifrnd(lowerb,upperb,[l1_neurons 1]);
w=unifrnd(lowerb,upperb,[l2_neurons l1_neurons]);


error_train=zeros(num_of_train,1);
error_test=zeros(num_of_test,1);

output_train=zeros(num_of_train,1);
output_test=zeros(num_of_test,1);

mse_train=zeros(max_epoch_p,1);
mse_test=zeros(max_epoch_p,1);
%% 2 layer perceptron
% Train
for i=1:max_epoch_p
  for j=1:num_of_train
      
    input=data_train(j,1:m-1);
    target=data_train(j,m);
    
    % Encoder1
    net_e1=sum((input - mean1).^2, 2);
    h1=exp(-0.5 * net_e1./(sigma1.^2));
        
    % Encoder2
    net_e2=sum((h1' - mean2).^2, 2);
    h2=exp(-0.5 * net_e2./(sigma2.^2));

    % Layer 1
    net11=sum((h2' - mean3).^2, 2);  
    o1=exp(-0.5 * net11./(sigma3.^2));

    % Layer 2
    nett22=w*o1;  
    o2=nett22;  

    % Predicted  Train Output
    output_train(j,1)=o2;

    % Calc Error Train
    e=target-o2; 
    error_train(j,1)=e;

    % Back Propagation
      delta_mean3 = eta_m3*e*-1*w'* ((h2'- mean3)'*(o1./(sigma3.^2)))';  
      delta_sigma3=eta_s3*e*-1*w*sum((h2' - mean3).^2, 2).*(1./(sigma3.^3)).*o1;  
      mean3=mean3-delta_mean3;
      sigma3=sigma3-delta_sigma3;
      w=w-eta_w3*e'*-1*1*o1';

  end 


     % c1 square train error
     mse_train(i,1)=mse(error_train);
     
  %% Test
     for j=1:num_of_test
        
         input=data_test(j,1:m-1);
         target=data_test(j,m);
         
         % Feedforward
        
        
    % Encoder1
    net_e1=sum((input - mean1).^2, 2);
    h1=exp(-0.5 * net_e1./(sigma1.^2));
        
    % Encoder2
    net_e2=sum((h1' - mean2).^2, 2);
    h2=exp(-0.5 * net_e2./(sigma2.^2));

    % Layer 1
    net11=sum((h2' - mean3).^2, 2);  
    o1=exp(-0.5 * net11./(sigma3.^2));

    % Layer 2
    nett22=w*o1;  
    o2=nett22;  
        
          % Predicted output
          output_test(j,1)=o2;
         
          % Calc error
          e=target-o2;
          error_test(j,1)=e;
%          
     end
      % c1 square test error
     mse_test(i,1)=mse(error_test);
     
  %% Find Regression
  [m_train ,b_train]=polyfit(data_train(:,m),output_train(:,1),1);
  [y_fit_train,~] = polyval(m_train,data_train(:,m),b_train);
  [m_test ,b_test]=polyfit(data_test(:,m),output_test(:,1),1);
  [y_fit_test,~] = polyval(m_test,data_test(:,m),b_test);
 
 %% plot results
      figure(1)
      subplot(2,3,1)
      plot(data_train(:,m),'-r')
      hold on
      subplot(2,3,1)
      plot(output_train,'b')
      title('Output Train')
      hold off;
 
      subplot(2,3,2)
      semilogy(mse_train(1:i,1),'-r')
      title('MSE Train')
      hold off;
      
      subplot(2,3,3)
      plot(data_train(:,m),output_train(:,1),'b*')
      hold on
      plot(data_train,data_train,'r')
      title('Regression train')
      hold off;
 
      
      subplot(2,3,4)
      plot(data_test(:,m),'-r')
      hold on
      subplot(2,3,4)
      plot(output_test,'b')
      title('Output test')
      hold off;
 
      subplot(2,3,5)
      semilogy(mse_test(1:i,1),'-r')
      title('MSE test')
      hold off;
      
      subplot(2,3,6)
      plot(data_test(:,m),output_test(:,1),'b*')
      hold on
      plot(data_test,data_test,'r')
      title('Regression test')
      hold off;
end
 fprintf('mse train = %1.16g, mse test = %1.16g \n', mse_train(max_epoch_p,1), mse_test(max_epoch_p,1))
