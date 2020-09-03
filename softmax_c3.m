function softmax_result= softmax_c3( input )

[m,n,l]=size(input);

% r_input=reshape(permute(input,[3,1,2]),[l,m*n]);

% max_input=max(r_input);

for i=1:l
    softmax_result(:,:,i)=exp(input(:,:,i))./sum(exp(input),3); 
end


end

