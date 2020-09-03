clear
close all
clc

file_name=dir('caffe_result/*mat');

angle=0:30:330;

t1=1;
t2=0.5;
t3=1;
t4=1;
t5=0.5;

for i=1:162
    seg=load(['caffe_result/',file_name(i).name]);
    
    name=file_name(i).name;
    name=name(1:end-4);
    
    im=imread(['original_image/',name,'.jpg']);
    [m,n,c]=size(im);
    all_c=zeros(513,513,5);
    
    for j=1:12
        for k=1:5
            c=softmax_c3(seg.caffe_result{j}{k});
            im=padarray(c(:,:,2),[250 250]);
            r_im = im_rotation(im1, 513, 360+angle(j));
            r_im = permute(r_im,[2,1,3]);
            all_c(:,:,k)=all_c(:,:,k)+r_im;
        end
    end
    
    f_c=imresize(all_c,[m,n]);
    
    f_c1=f_c(:,:,1);
    f_c2=f_c(:,:,2);
    f_c3=f_c(:,:,3);
    f_c4=f_c(:,:,4);
    f_c5=f_c(:,:,5);
  
    f_c1(f_c1>t1)=1;
    f_c1(f_c1~=1)=0;
    
    f_c2(f_c2>t2)=1;
    f_c2(f_c2~=1)=0;   
    
    f_c3(f_c3>t3)=1;
    f_c3(f_c3~=1)=0;
    
    f_c4(f_c4>t4)=1;
    f_c4(f_c4~=1)=0;
    
    f_c5(f_c5>t5)=1;
    f_c5(f_c5~=1)=0;
      
    tiff(:,:,1)=f_c1;
    tiff(:,:,2)=f_c2;
    tiff(:,:,3)=f_c3;
    tiff(:,:,4)=f_c4;
    tiff(:,:,5)=f_c5;    
    
    for j=1:5
        tiff(:,:,j)=imfill(tiff(:,:,j),'hole');
    end

    saveastiff(uint8(tiff),['/binary result',name,'.tif'])
    clear tiff 
end