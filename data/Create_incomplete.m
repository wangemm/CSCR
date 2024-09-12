data_name = 'corel5k_six_view';
data_path = [data_name '.mat'];
load(data_path);
sample_mask_ratio = 0.5;
label_mask_ratio = 0.5;
train_data_ratio = 0.9;
num_iterations = 30;
new_data = cellfun(@(x) double(x), X.', 'UniformOutput', false);
folds_data = cell(1, num_iterations);
folds_label = cell(1, num_iterations);
folds_sample_index = cell(1, num_iterations);
for i = 1:num_iterations
    %% 设置不同的随机数种子（使用循环变量i作为种子）
    rng(i);
    %% create incomplete views
    [X,W] = Mark_featureN(new_data,sample_mask_ratio,1);
    num_matrices = numel(W);
    num_rows = size(W{1}, 1);
    new_W = zeros(num_rows, num_matrices);
    for ii = 1:num_matrices
        matrix = W{ii};
        new_W(:, ii) = matrix(:, 1);
    end 
    [n_data,n_label] = size(label);
    train_num = ceil(n_data*train_data_ratio);
    index_perm = randperm(n_data);
    train_index = index_perm(1,1:train_num);
    test_index = index_perm(1,train_num+1:end);
    %% create missing labels
    label=full(label);
    obrT = Mark_label(label,train_index,label_mask_ratio);
    
    if  min(min(label))==-1
        label=(label+1)/2;
    end
    folds_data{i} = new_W;
    folds_label{i} = obrT;
    folds_sample_index{i} = index_perm;
end
disp('Complete');
save('corel5k_six_view_MaskRatios_0.5_LabelMaskRatio_0.5_TraindataRatio_0.9.mat', 'folds_data', 'folds_label', 'folds_sample_index');
