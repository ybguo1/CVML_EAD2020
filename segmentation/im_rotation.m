function center_patch = im_rotation(im, patch_size, angle)

im=double(im);
[m,n,c]=size(im);
center=(m+1)/2;
angle=angle/180*pi;

[x,y]   = meshgrid(1:m,1:m);
rotation_matrix = [cos(angle) sin(angle);-sin(angle) cos(angle)];
rotatedcoords = rotation_matrix*[x(:)-center,y(:)-center]';

for i=1:3
    rpatch = interp2(im(:,:,i), rotatedcoords(1,:)+center,rotatedcoords(2,:)+center, 'linear');
    rim= reshape(rpatch,[m,m]);
    center_patch(:,:,i)=rim(center-(patch_size-1)/2:center+(patch_size-1)/2,center-(patch_size-1)/2:center+(patch_size-1)/2);
end

