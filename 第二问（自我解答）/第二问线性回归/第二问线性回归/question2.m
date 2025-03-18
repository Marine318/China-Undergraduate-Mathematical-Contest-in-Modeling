 
 % 定义文件路径和数据集名称
file_paths = {'各品类均利率.xlsx', '各品类销售量.xlsx'};
dataset_names = {'花菜类', '花叶类', '辣椒类', '茄类', '食用菌类', '水生根茎类'};
data1=readtable("各品类均利率.xlsx");
data1.Properties.VariableNames(1) = "date";
data1.Properties.VariableNames(2) = "veg1";
data1.Properties.VariableNames(3) = "veg2";
data1.Properties.VariableNames(4) = "veg3";
data1.Properties.VariableNames(5) = "veg4";
data1.Properties.VariableNames(6) = "veg5";
data1.Properties.VariableNames(7) = "veg6";
datasale=readtable("各品类销售量.xlsx");
datasale.Properties.VariableNames(1) = "date";
datasale.Properties.VariableNames(2) = "veg1";
datasale.Properties.VariableNames(3) = "veg2";
datasale.Properties.VariableNames(4) = "veg3";
datasale.Properties.VariableNames(5) = "veg4";
datasale.Properties.VariableNames(6) = "veg5";
datasale.Properties.VariableNames(7) = "veg6";

% 循环处理每组数据
for i = 1:6
    % 导入数据
    
    
    % 提取销售量和利率
    sale_data = datasale.(['veg' num2str(i)]);
    rate_data = data1.(['veg' num2str(i)]);
    
    % 找出非零的数据点的索引
    nonzero_indices = (rate_data ~= 0);
    
    % 使用逻辑索引筛选非零的数据点
    x_nonzero = sale_data(nonzero_indices);
    y_nonzero = rate_data(nonzero_indices);
    
    % 异常值处理（三倍标准差法）
    mean_y = mean(y_nonzero);
    std_y = std(y_nonzero);
    threshold = 3*std_y;
    outliers = abs(y_nonzero - mean_y) > threshold;
    
    mean_x = mean(x_nonzero);
    std_x = std(x_nonzero);
    threshold = 3*std_x;
    outliers = abs(x_nonzero - mean_x) > threshold;
    % 将异常值更改为中位数
    y_nonzero(outliers) = median(y_nonzero);
    x_nonzero(outliers) = median(x_nonzero);
    % 执行一元线性回归
    coefficients = polyfit(x_nonzero, y_nonzero, 1);
    
    % 提取斜率和截距
    slope = coefficients(1);
    intercept = coefficients(2);
    
    % 绘制原始数据点和回归线
    figure;
    scatter(x_nonzero, y_nonzero, 'b', 'filled');
    hold on;
    x_range = min(x_nonzero):0.01:max(x_nonzero);
    y_fit = slope * x_range + intercept;
    plot(x_range, y_fit, 'r', 'LineWidth', 2);
    legend('原始数据', '线性回归');
    xlabel(['销售量' num2str(i)]);
    ylabel(['利率' num2str(i)]);
    title(['线性回归' dataset_names{i}]);
    grid on;
    % 在执行一元线性回归后，获取回归系数
    % 在执行一元线性回归后，获取回归系数
 

    % 计算残差
    y_fit = slope * x_nonzero + intercept;
    residuals = y_nonzero - y_fit;
    
    % 计算总平方和
    total_sum_of_squares = sum((y_nonzero - mean(y_nonzero)).^2);
    
    % 计算残差平方和
    residual_sum_of_squares = sum(residuals.^2);
    
    % 计算自变量数量
    num_predictors = 1;  % 如果有多个自变量，根据实际情况更改
    
    % 计算调整的 R-squared 值
    n = length(y_nonzero);  % 样本数量
    adjusted_r_squared = 1 - ((n - 1) / (n - num_predictors - 1)) * (residual_sum_of_squares / total_sum_of_squares);
    
    % 输出调整的 R-squared 值
    fprintf('调整的 R-squared 值：%f\n', adjusted_r_squared);
    fprintf('a=%f,b=%f\n',slope,intercept)
    
        
        % 在这里可以根据需要进一步汇总或保存回归结果
    end
    

