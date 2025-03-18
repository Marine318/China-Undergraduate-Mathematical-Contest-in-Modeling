% 初始化种群
for i = 1:33
    c=xlsread('第三问预测数据','c2:c34');
    f=[19.74514286
17.23542857
10.80185714
8.857142857
7.142857143
38.29585714
10.145
19.05328571
7
6.337285714
29.14285714
19.57585714
25.01857143
11.28571429
9.865714286
21.10271429
16.57571429
14.508
15.89185714
19.85714286
5.252571429
14.28571429
55.24685714
63.26014286
4.667142857
19.25428571
26.02128571
25.14285714
9.641714286
12.28571429
3.780571429
50.73128571
]
    
    
    
f = @(x1, x2) ((1+x1)*c(i)*f-x2*c(i)); % 新的目标函数，最大化 x1^2 + x2^2

figure(1);
[X1, X2] = meshgrid(linspace(2.5, 30, 100), linspace(0, 2, 100));
Z = (1+x1)*c(i)*f-x2*c(i);
contour(X1, X2, Z, 50);

N = 50;                    % 初始种群个数
d = 2;                     % 空间维数
ger = 100;                 % 最大迭代次数
limit = [-5, 5; -5, 5];    % 设置位置参数限制
vlimit = [-1, 1; -1, 1];   % 设置速度限制
w = 0.8;                   % 惯性权重
c1 = 0.5;                  % 自我学习因子
c2 = 0.5;                  % 群体学习因子

% 随机初始化种群的位置
x = zeros(N, d);
for i = 1:d
    x(:, i) = limit(i, 1) + (limit(i, 2) - limit(i, 1)) * rand(N, 1);
end

% 随机初始化种群的速度
v = rand(N, d);
for i = 1:d
    v(:, i) = vlimit(i, 1) + (vlimit(i, 2) - vlimit(i, 1)) * rand(N, 1);
end

xm = x;                % 每个个体的历史最佳位置
ym = zeros(1, d);      % 种群的历史最佳位置
fxm = zeros(N, 1);     % 每个个体的历史最佳适应度
fym = -inf;            % 种群历史最佳适应度

% 绘制初始状态图
hold on;
plot(xm(:, 1), xm(:, 2), 'ro');
title('初始状态图');
xlabel('x1');
ylabel('x2');
grid on;

figure(2);
% 群体更新
iter = 1;
record = zeros(ger, 1);  % 记录器
while iter <= ger
    fx = -(x(:, 1).^2 + x(:, 2).^2);  % 计算个体当前适应度，负号是因为我们要最大化目标函数
    
    % 更新个体历史最佳适应度和位置
    for i = 1:N
        if fxm(i) < fx(i)
            fxm(i) = fx(i);
            xm(i, :) = x(i, :);
        end
    end
    
    % 更新群体历史最佳适应度和位置
    if fym < max(fxm)
        [fym, nmax] = max(fxm);
        ym = xm(nmax, :);
    end
    
    % 速度更新
    v = w * v + c1 * rand * (xm - x) + c2 * rand * (repmat(ym, N, 1) - x);
    
    % 边界速度处理
    for i = 1:d
        v(v(:, i) > vlimit(i, 2), i) = vlimit(i, 2);
        v(v(:, i) < vlimit(i, 1), i) = vlimit(i, 1);
    end
    
    % 位置更新
    x = x + v;
    
    % 边界位置处理
    for i = 1:d
        x(x(:, i) > limit(i, 2), i) = limit(i, 2);
        x(x(:, i) < limit(i, 1), i) = limit(i, 1);
    end
    
    % 记录最大值
    record(iter) = -fym;  % 注意取负号还原最大值
    
    % 绘制状态位置变化图
    clf;
    contour(X1, X2, Z, 50);
    hold on;
    plot(xm(:, 1), xm(:, 2), 'ro');
    title('状态位置变化');
    xlabel('x1');
    ylabel('x2');
    grid on;
    pause(0.1);
    
    iter = iter + 1;
end

% 绘制收敛过程图
figure(3);
plot(record);
title('收敛过程');

% 绘制最终状态位置图
figure(4);
contour(X1, X2, Z, 50);
hold on;
plot(xm(:, 1), xm(:, 2), 'ro');
title('最终状态位置');
xlabel('x1');
ylabel('x2');
grid on;

disp(['最大值：', num2str(-fym)]);  % 注意取负号还原最大值
disp(['变量取值：x1 = ', num2str(ym(1)), ', x2 = ', num2str(ym(2))]);
end