% 遗传算法参数设置
options = gaoptimset('PopulationSize', 50, 'Generations', 100, 'TolFun', 1e-6);

% 定义目标函数（包括惩罚项）
fitnessFunction = @(x) -obj_function(x);

% 定义不等式约束条件
nonlcon = @(x) constraints(x);

% 定义变量的下界和上界
lb = [0; 0];  % 变量下界
ub = [Inf; Inf];  % 变量上界

% 使用遗传算法求解带不等式约束的优化问题
[x, fval] = ga(fitnessFunction, 2, [], [], [], [], lb, ub, nonlcon, options);

disp('最优解为：');
disp(x);
disp('目标表达式的最大值等于：');
disp(-fval);

% 定义目标函数（包括惩罚项）
function [obj] = obj_function(x)
    % 目标函数
    obj = 6.639983587*(-2.636564*x(1)+63.0362159)*(1+x(1))-x(2)*6.639983587;
end

% 定义不等式约束条件
function [c, ceq] = constraints(x)
    % 不等式约束条件
    c1 = (-2.636564*x(1)+63.0362159 )*(1-0.94376296)- x(2); % x1^2 - x2 >= 0
    c3=x(2);
     c2 =(1+x(1))*6.639983587-34; % -x1 - x2^2 + 2 <= 0
     c4=(1+x(1))*6.639983587;
     
    % 等式约束为空
    ceq = [];
    
    % 将不等式约束的值放入向量 c 中
    c = [c1; -c2;c3;c4]; % 注意不等式约束的方向取反
end