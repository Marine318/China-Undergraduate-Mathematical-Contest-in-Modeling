clc;clear;close all;	
load('C2_predict.mat')	
	
data_str='代码数据.xlsx' ;  %读取数据的路径 	
	
	
data1=readtable(data_str,'VariableNamingRule','preserve'); %读取数据 	
data2=data1(:,2:end); 	
data=table2array(data1(:,2:end));	
data_biao=data2.Properties.VariableNames;  %数据特征的名称	
	
  A_data1=data;	
 data_biao1=data_biao;	
 select_feature_num=G_out_data.select_feature_num1;   %特征选择的个数	
	
data_select=A_data1;	
feature_need_last=1:size(A_data1,2)-1;	
	
	
	
%% 数据划分	
  x_feature_label=data_select(:,1:end-1);    %x特征	
  y_feature_label=data_select(:,end);          %y标签	
 index_label1=1:(size(x_feature_label,1));	
 index_label=G_out_data.spilt_label_data;  % 数据索引	
 if isempty(index_label)	
     index_label=index_label1;	
 end	
spilt_ri=G_out_data.spilt_rio;  %划分比例 训练集:验证集:测试集	
train_num=round(spilt_ri(1)/(sum(spilt_ri))*size(x_feature_label,1));          %训练集个数	
vaild_num=round((spilt_ri(1)+spilt_ri(2))/(sum(spilt_ri))*size(x_feature_label,1)); %验证集个数	
 %训练集，验证集，测试集	
 train_x_feature_label=x_feature_label(index_label(1:train_num),:);	
 train_y_feature_label=y_feature_label(index_label(1:train_num),:);	
 vaild_x_feature_label=x_feature_label(index_label(train_num+1:vaild_num),:);	
vaild_y_feature_label=y_feature_label(index_label(train_num+1:vaild_num),:);	
 test_x_feature_label=x_feature_label(index_label(vaild_num+1:end),:);	
 test_y_feature_label=y_feature_label(index_label(vaild_num+1:end),:);	
%Zscore 标准化	
%训练集	
 x_mu = mean(train_x_feature_label);  x_sig = std(train_x_feature_label); 	
 train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化	
 y_mu = mean(train_y_feature_label);  y_sig = std(train_y_feature_label); 	
train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化  	
%验证集	
 vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;    %验证数据标准化	
 vaild_y_feature_label_norm=(vaild_y_feature_label - y_mu) ./ y_sig;  %验证数据标准化	
%测试集	
test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;    % 测试数据标准化	
 test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化  	
	
%% 参数设置	
num_pop=G_out_data.num_pop1;   %种群数量	
num_iter=G_out_data.num_iter1;   %种群迭代数	
method_mti=G_out_data.method_mti1;   %优化方法	
BO_iter=G_out_data.BO_iter;   %贝叶斯迭代次数	
min_batchsize=G_out_data.min_batchsize;   %batchsize	
max_epoch=G_out_data.max_epoch1;   %maxepoch	
hidden_size=G_out_data.hidden_size1;   %hidden_size	
	
	
%% 算法处理块	
	
	
	
 hidden_size=G_out_data.hidden_size1;    %神经网络隐藏层	
disp('BiLSTM回归')	
t1=clock; 	
 max_epoch=G_out_data.max_epoch1;    %神经网络隐藏层	
for i = 1: size(train_x_feature_label,1)      %修改输入变成元胞形式	
    p_train1{i, 1} = (train_x_feature_label_norm(i,:))';	
end	
for i = 1 : size(test_x_feature_label,1)	
     p_test1{i, 1}  = (test_x_feature_label_norm(i,:))';	
	
 end	
	
 for i = 1 : size(vaild_x_feature_label,1)	
     p_vaild1{i, 1}  = (vaild_x_feature_label_norm(i,:))';	
 end	
	
	
	
	
 [Mdl,fitness] = optimize_fitrBiLSTM(p_train1,train_y_feature_label_norm,p_vaild1,vaild_y_feature_label_norm,num_pop,num_iter,method_mti,max_epoch,min_batchsize);         	
	
y_train_predict_norm = predict(Mdl, p_train1,'MiniBatchSize',min_batchsize);	
	
 y_vaild_predict_norm = predict(Mdl, p_vaild1,'MiniBatchSize',min_batchsize);	
 y_test_predict_norm =  predict(Mdl, p_test1,'MiniBatchSize',min_batchsize);	
 t2=clock;	
 Time=t2(3)*3600*24+t2(4)*3600+t2(5)*60+t2(6)-(t1(3)*3600*24+t1(4)*3600+t1(5)*60+t1(6));	
	
	
graph= layerGraph(Mdl.Layers); figure; plot(graph) 	
analyzeNetwork(Mdl)	
	
	
 y_train_predict=y_train_predict_norm*y_sig+y_mu;  %反标准化操作 	
 y_vaild_predict=y_vaild_predict_norm*y_sig+y_mu; 	
 y_test_predict=y_test_predict_norm*y_sig+y_mu; 	
 train_y=train_y_feature_label; disp('***************************************************************************************************************')   	
train_MAE=sum(abs(y_train_predict-train_y))/length(train_y) ; disp(['训练集平均绝对误差MAE：',num2str(train_MAE)])	
train_MAPE=sum(abs((y_train_predict-train_y)./train_y))/length(train_y); disp(['训练集平均相对误差MAPE：',num2str(train_MAPE)])	
train_MSE=(sum(((y_train_predict-train_y)).^2)/length(train_y)); disp(['训练集均方根误差MSE：',num2str(train_MSE)]) 	
 train_RMSE=sqrt(sum(((y_train_predict-train_y)).^2)/length(train_y)); disp(['训练集均方根误差RMSE：',num2str(train_RMSE)]) 	
train_R2= 1 - (norm(train_y - y_train_predict)^2 / norm(train_y - mean(train_y))^2);   disp(['训练集均方根误差R2：',num2str(train_R2)]) 	
vaild_y=vaild_y_feature_label;disp('***************************************************************************************************************')	
vaild_MAE=sum(abs(y_vaild_predict-vaild_y))/length(vaild_y) ; disp(['验证集平均绝对误差MAE：',num2str(vaild_MAE)])	
vaild_MAPE=sum(abs((y_vaild_predict-vaild_y)./vaild_y))/length(vaild_y); disp(['验证集平均相对误差MAPE：',num2str(vaild_MAPE)])	
vaild_MSE=(sum(((y_vaild_predict-vaild_y)).^2)/length(vaild_y)); disp(['验证集均方根误差MSE：',num2str(vaild_MSE)])     	
 vaild_RMSE=sqrt(sum(((y_vaild_predict-vaild_y)).^2)/length(vaild_y)); disp(['验证集均方根误差RMSE：',num2str(vaild_RMSE)]) 	
vaild_R2= 1 - (norm(vaild_y - y_vaild_predict)^2 / norm(vaild_y - mean(vaild_y))^2);    disp(['验证集均方根误差R2:  ',num2str(vaild_R2)])			
 test_y=test_y_feature_label;disp('***************************************************************************************************************');   	
test_MAE=sum(abs(y_test_predict-test_y))/length(test_y) ; disp(['测试集平均绝对误差MAE：',num2str(test_MAE)])        	
test_MAPE=sum(abs((y_test_predict-test_y)./test_y))/length(test_y); disp(['测试集平均相对误差MAPE：',num2str(test_MAPE)])	
 test_MSE=(sum(((y_test_predict-test_y)).^2)/length(test_y)); disp(['测试集均方根误差MSE：',num2str(test_MSE)]) 	
 test_RMSE=sqrt(sum(((y_test_predict-test_y)).^2)/length(test_y)); disp(['测试集均方根误差RMSE：',num2str(test_RMSE)]) 	
 test_R2= 1 - (norm(test_y - y_test_predict)^2 / norm(test_y - mean(test_y))^2);   disp(['测试集均方根误差R2：',num2str(test_R2)]) 	
disp(['算法运行时间Time: ',num2str(Time)])	
%% 绘图块	
color_list=G_out_data.color_list;   %颜色数据库	
rand_list1=G_out_data.rand_list1;   %颜色数据库	
Line_Width=G_out_data.Line_Width;   %线粗细	
makesize=G_out_data.makesize;   %标记大小	
yang_str2=G_out_data.yang_str2;   %符号库	
yang_str3=G_out_data.yang_str3;   %符号库	
kuang_width=G_out_data.kuang_width;   %画图展示数据	
show_num=G_out_data.show_num;   %测试集画图展示数据	
show_num1=G_out_data.show_num1;   %验证集画图展示数据	
show_num2=G_out_data.show_num2;   %训练集画图展示数据	
	
FontSize=G_out_data.FontSize;   %字体设置	
xlabel1=G_out_data.xlabel1;   %	
ylabel1=G_out_data.ylabel1;   %	
title1=G_out_data.title1;   %	
legend1=G_out_data.legend1;   %图例	
box1=G_out_data.box1;   %框	
le_kuang=G_out_data.le_kuang;   %图例框	
grid1=G_out_data.grid1;   %网格	
figure	
XX=1:length(train_y_feature_label);	
 index_show=1:show_num2;	
plot(gca,XX(index_show),train_y_feature_label(index_show),yang_str2{1,3},'Color',color_list(rand_list1(1),:),'LineWidth',Line_Width(1))	
hold (gca,'on')	
 plot(gca, XX(index_show),y_train_predict(index_show),yang_str3{1,1},'Color',color_list(rand_list1(2),:),'LineWidth',Line_Width(2),'MarkerSize',makesize)	
hold (gca,'on')	
 set(gca,'FontSize',FontSize,'LineWidth',kuang_width)	
 xlabel(gca,xlabel1)	
 ylabel(gca,ylabel1)	
 title(gca,'训练集结果')	
 legend(gca,legend1) 	
  box(gca,box1)	
 legend(gca,le_kuang) %图例框消失	
 grid(gca,grid1)	
	
	
figure	
XX=1:length(vaild_y_feature_label);	
 index_show=1:show_num1;	
plot(gca,XX(index_show),vaild_y_feature_label(index_show),yang_str2{1,3},'Color',color_list(rand_list1(1),:),'LineWidth',Line_Width(1))	
hold (gca,'on')	
 plot(gca, XX(index_show),y_vaild_predict(index_show),yang_str3{1,1},'Color',color_list(rand_list1(2),:),'LineWidth',Line_Width(2),'MarkerSize',makesize)	
hold (gca,'on')	
 set(gca,'FontSize',FontSize,'LineWidth',kuang_width)	
 xlabel(gca,xlabel1)	
 ylabel(gca,ylabel1)	
 title(gca,'验证集结果')	
 legend(gca,legend1) 	
  box(gca,box1)	
 legend(gca,le_kuang) %图例框消失	
 grid(gca,grid1)	
	
	
figure	
XX=1:length(test_y_feature_label);	
 index_show=1:show_num;	
plot(gca,XX(index_show),test_y_feature_label(index_show),yang_str2{1,3},'Color',color_list(rand_list1(1),:),'LineWidth',Line_Width(1))	
hold (gca,'on')	
 plot(gca, XX(index_show),y_test_predict(index_show),yang_str3{1,1},'Color',color_list(rand_list1(2),:),'LineWidth',Line_Width(2),'MarkerSize',makesize)	
hold (gca,'on')	
 set(gca,'FontSize',FontSize,'LineWidth',kuang_width)	
 xlabel(gca,xlabel1)	
 ylabel(gca,ylabel1)	
 title(gca,'测试集结果')	
 legend(gca,legend1) 	
  box(gca,box1)	
 legend(gca,le_kuang) %图例框消失	
 grid(gca,grid1)	
	
