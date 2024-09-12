    function [data,W] = Mark_featureN(data,percent,a)
    %MARK_FEATURE 此处显示有关此函数的摘要
    %   此处显示详细说明
    % data is m*1 cell with a matrix of size n*dt in each cell
    % percent is the mask_ratio of missing samples
    [n,~]=size(data{1});
    m=length(data);
    W=cell(m,1);
    index_delete=ones(n,m);
    indicator=cell(m,1);
    for ii=1:m
        if ii~=m
% % % % % % %         temp=ones(n,1);
% %         temp(:)=1;
        index=randperm(n);
        Mask_index=index(1,1:floor(n*percent));
        index_delete(Mask_index,ii)=0;
% % % % % % %         temp(Mask_index,1)=percent;
% % % % % % %         %    temp(setdiff(index,Mask_index),1)=1-percent;
% % % % % % %         W{ii}=diag(temp);
% % % % % % %         data{ii}(Mask_index,:)=0;
% % % % % % %         %             indicator{ii}=isnan(te);
        indicator{ii}=Mask_index;
% % % % % % % 
% % % % % % %         % estimiating missing value
% % % % % % %         data{ii}(Mask_index,:)=repmat(mean(data{ii},1),length(Mask_index),1);
        else
            temp=index_delete(:,1:ii-1);
            zeros_sample=find(sum(temp,2)==0);
            M_sample=setdiff(1:n,zeros_sample);% 找到还可以继续删除视图的样本
            M=length(M_sample);
            index=randperm(M);
            M_select=index(1,1:floor(n*percent));
            Mask_index=M_sample(1,M_select);
            index_delete(Mask_index,ii)=0;
            indicator{ii}=Mask_index;
        end
    end
    %% estimiate missing values in matrix.
    % estimiating missing values using average value
    for iii=1:m
% %         temp=ones(n,1);
% %         temp(indicator{iii},1)=1-percent;
% %         W{iii}=diag(temp);
        data{iii}(indicator{iii},:)=0;
        W{iii}=ones(size(data{iii}));
        W{iii}(indicator{iii},:)=0; % return the missing index here
        
        if (a==1)
        view_1=setdiff(1:n,indicator{iii});
        data_temp=data{iii}(view_1,:);
        temp=abs(data_temp);
        data{iii}(view_1,:)=temp./repmat(sum(temp,2),1,size(temp,2));
        end
        clum=sum(data{iii},1);
        del_clum=find(clum==0);
        data{iii}(:,del_clum)=[];
        W{iii}(:,del_clum)=[];
    end
% %     if(a==1)
% %         %% normalize feature for each view
% %         for mm=1:length(data)
% %             temp=abs(data{mm});
% %             newdata{mm}=temp./repmat(sum(temp,2),1,size(temp,2));
% %         end
% %     end
% %     temp_delete=sum(index_delete,2);
% %     sample_delete=find(temp_delete==0);
% % % %     indicator{m+1,1}=sample_delete;
% %     data=newdata;


    end

