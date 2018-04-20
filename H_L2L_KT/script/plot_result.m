function [ output_args ] = plot_result( dataset )
%PLOT_RESULT Summary of this function goes here
%   Detailed explanation goes here

done_file = strrep(dataset,'.mat','_done.mat');
load(done_file);

plot( mean(mean(no_acc,3)), '-b','LineWidth',2);  %no transfer
hold on
plot( mean(mean(pr_acc,3))/100, '-ko','LineWidth',2, 'MarkerSize',10); %prior-feat
hold on
plot( mean(mean(da_acc,3))/100, '-xm','LineWidth',2, 'MarkerSize',10); %svm-das
hold on

legend('No-Transfer','Prior-Features','H-L2L(SVM-DAS)','Location','SouthEast');

grid on
ylim([0.5 1]) 
xlim([0 10])

xlabel('# of positive training samples','FontSize' , 22 ,  'FontWeight', 'bold');
ylabel('Recognition Rate','FontSize' , 22 ,  'FontWeight', 'bold');
set(gca,'FontSize' , 12 ,  'FontWeight', 'bold');
title('Classification Accuracy Result','FontSize' , 24 ,  'FontWeight', 'bold');

end

