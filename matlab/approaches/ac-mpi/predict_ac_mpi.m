function label_pre = predict_ac_mpi(feat, nClass, nSamples, nfeatures)
%PREDICT_GDL �˴���ʾ�йش˺����ժҪ
%   �˴���ʾ��ϸ˵��
% a = 100 for USPS
% z = 0.01;
K = 20;
a = 1;
z = 0.01;
%LASTN = maxNumCompThreads(1);
%data_row = single(py.array.array('d'));  % Add order='F' to get data in column-major order (as in Fortran 'F' and Matlab
%data_size = cell2mat(cell(feat.shape));
%data = reshape(data_row, data_size);  % No need for transpose, since we're retrieving the data in column major order

data = double(py.array.array('d',py.numpy.nditer(feat))); %d is for double, see link below on types
data = reshape(data, [nfeatures nSamples])'; %Could incorporate x.shape here ...

feat = data;

distance_matrix = pdist2(feat, feat);
distance_matrix = distance_matrix.^2;
% path intergral
label_pre = gacCluster(distance_matrix, nClass, 'path', K, a, z);

end

