%两个连续型变量（各蔬菜品类相关性分析）相关性分析
%%%%%%%%%%%%%%%%相关性热图%%%%%%%%%%%%%%%%%%
% 创建一个示例数据集
data = xlsread('时间序列处理结果后相关性草稿.xlsx'); % 生成一个100x5的随机数据集
% 计算相关性矩阵
correlationMatrix = corr(data); % 使用corr函数计算相关性矩阵
figure
% 创建相关性矩阵的可视化图
corrplot(correlationMatrix, 'type', 'Pearson'); % 使用'Pearson'类型计算相关性
title('相关性热图'); % 设置图标题