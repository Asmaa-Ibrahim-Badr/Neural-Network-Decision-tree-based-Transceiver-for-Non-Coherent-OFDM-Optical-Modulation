iterations_test=100;% size of trained  data 
subcarriers=15;
%%%generated binary bits with 8 subcarriers 
M = 4;
bitsPerSym = log2(M);
input=randi([0,1],subcarriers*bitsPerSym,iterations_test);
%qam modulator
signal =qammod(input,M,'bin','InputType','bit');
subcarriers_total=2*(subcarriers+1);
sym=size(signal);  %%%%%%%%symbol number
%%%%%optical signal  
opt_signal=[zeros(1,sym(2));signal;zeros(1,sym(2));conj(flip(signal,1))];    %%32 row ,100column                                   
%%%%%%%%%ifft 
ifft_sig=ifft(opt_signal);
ifft_sig_unclipped=ifft_sig; %% target ouput 
%%% input for the nural network to predict the unclipped signal 
ifft_sig(find(ifft_sig<0))=0;
ifft_sig1=ifft_sig'; %%% clipped signal as input for  neural network
%%%%%%%%%%%%%predict the clipped parts using the neural netwrok
y=predict(trees_bagged,ifft_sig1);
fft_signal=fft(y);
Data_subcarriers=fft_signal(2:(subcarriers+1),:);
signal_out = qamdemod(Data_subcarriers,M,'bin','OutputType','bit');
[NUMBER_errore,RATIO] = biterr(signal_out,input);
BER=sum(sum(signal_out~=input))/(subcarriers*M*iterations_test);
%%%%%%%%%%%% Root mean square error between predicted output and the input
RMSE=sqrt(sum((y.'-ifft_sig_unclipped).^2))/size(y,1);
%%%plot Root mean square error
% set(gcf,'color','w');
% axes('FontSize',14)
% plot(BER,'-.')
% xlabel('Iterations','fontsize',14,'fontweight','b')
% ylabel('RMSE','fontsize',14,'fontweight','b')
% title('Root mean square error with 10 neural nodes')
% grid on 
% figure
% subplot(3,1,1)
% stem(y(1,:))
% subplot(3,1,2)
% stem(ifft_sig_unclipped(:,1))
% subplot(3,1,3)
% stem(ifft_sig1(:,1))

