iterations=100000;% size of trained  data 
subcarriers=3;% size of input data 
subcarriers_total=2*(subcarriers+1);
%qam modulator
M = 16;
bitsPerSym = log2(M);
input=randi([0,1],subcarriers*bitsPerSym,iterations);
signal =qammod(input,M,'bin','InputType','bit');
sym=size(signal);  %%%%%%%%symbol number
opt_signal=[zeros(1,sym(2));signal;zeros(1,sym(2));conj(flip(signal,1))];                                        %%%%%optical signal
%%%%%%%%% IFFT
ifft_sig_trained=ifft(opt_signal);
%%%%%%%%%%%%%clipping 
ifft_sig_unclipped1=ifft_sig_trained; %% target ouput 
Ind=find(ifft_sig_trained<0);
ifft_sig_trained(Ind)=0;
% x=ifft_sig_trained;%%% input for the neural network to predict the unclipped signal 
% ifft_sig1={x(1,:);x(2,:);x(3,:);x(4,:);x(5,:);x(6,:);x(7,:);x(8,:)};
% ifft_sig_unclipped_out={ifft_sig_unclipped1(1,:),ifft_sig_unclipped1(2,:),ifft_sig_unclipped1(3,:),ifft_sig_unclipped1(4,:),ifft_sig_unclipped1(5,:),ifft_sig_unclipped1(6,:),ifft_sig_unclipped1(7,:),ifft_sig_unclipped1(8,:)};
trees=cell(subcarriers_total,1);
configurer=cell(subcarriers_total,1);
x='C:\users\asmaa.ibrahim\Documents\MATLAB'
for i=1:subcarriers_total 
    trees{i}=fitrtree(ifft_sig_trained',(ifft_sig_unclipped1(i,:)));
    configurer{i} = learnerCoderConfigurer(trees{i},ifft_sig_trained');
%     cfg = coder.config('lib','ecoder',true);
%     cfg.CustomFileNameStr = 'ec_$N$i';
    b=strcat(x, strcat('\Subcarrier_', num2str(i)));
    generateCode(configurer{i},'outputpath',b)
end
% trained_net = net1;
% save trained_net_svm



    
