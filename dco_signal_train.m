iterations=10000;% size of trained  data 
subcarriers=3;% size of input data 
input=randi([0,15],subcarriers,iterations);
%qam modulator
signal =qammod(input,16);
sym=size(signal);  %%%%%%%%symbol number
opt_signal=[zeros(1,sym(2));signal;zeros(1,sym(2));conj(flip(signal,1))];                                        %%%%%optical signal
%%%%%%%%% IFFT
ifft_sig_trained=ifft(opt_signal,[],1);
%%%%%%%%%%%%%clipping 
ifft_sig_unclipped1=ifft_sig_trained; %% target ouput 
Ind=find(ifft_sig_trained<0);
ifft_sig_trained(Ind)=-1*ifft_sig_trained(Ind);
% x=ifft_sig_trained;%%% input for the neural network to predict the unclipped signal 
% ifft_sig1={x(1,:);x(2,:);x(3,:);x(4,:);x(5,:);x(6,:);x(7,:);x(8,:)};
% ifft_sig_unclipped_out={ifft_sig_unclipped1(1,:),ifft_sig_unclipped1(2,:),ifft_sig_unclipped1(3,:),ifft_sig_unclipped1(4,:),ifft_sig_unclipped1(5,:),ifft_sig_unclipped1(6,:),ifft_sig_unclipped1(7,:),ifft_sig_unclipped1(8,:)};
net1=N_network(ifft_sig_trained,ifft_sig_unclipped1);
trained_net2 = net1;
save trained_net2



    
