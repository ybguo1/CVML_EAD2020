clear
close all
clc

addpath('/media/localuser/99630c81-35b5-4ffb-a4ad-a23fdd8d3a3d/caffe-master/matlab');

%%%%% Model Path %%%%%
model_dir='./model/';

net_model= [model_dir 'deploy.prototxt'];
net_weights= [model_dir 'snapshot_iter_381960.caffemodel'];

%%%%% Load model %%%%%

disp('Loading model..')
phase = 'test'; 
caffe.set_mode_gpu();
gpu_id = 0;
caffe.set_device(gpu_id);
net= caffe.Net(net_model, net_weights, phase);
disp('Completed')

meandata(:,:,3)=12.3779773712;
meandata(:,:,2)=15.694311142;
meandata(:,:,1)=25.7220363617;

path='./test_data/EAD2020-Phase-II-Evaluation/SemanticSegmentation/';
name=dir([path,'/*jpg']);

angle=0:30:330;

for i=1:length(name)
    
        im=imread([path,name(i).name]);        
        im=imresize(im,[513,513]);
        
        im1=padarray(im,[250 250]);
        
        for j=1:length(angle)
            rpatch = im_rotation(im1, 513, angle(j));
            im=[rpatch,zeros(513,513*4,3)];
        
            im_name=name(i).name;
            im_name=im_name(1:end-4);
        
            imr=im;
            im_data = im(:, :, [3, 2, 1]); 
            im_data = permute(im_data, [2, 1, 3]);
            im_data = single(im_data); 
            im_data=im_data-single(meandata);
            processed_image={im_data};
            [m,n,c]=size(im);
            caffe_result{j}  = net.forward( processed_image );
        end
            save(['caffe_result/',im_name,'.mat'],'caffe_result')

end
