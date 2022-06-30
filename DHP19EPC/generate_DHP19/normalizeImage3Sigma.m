function normalizedMat = normalizeImage3Sigma(img)
    [m,n] = size(img);
    sum_img=sum(sum(img));
    count_img=sum(sum(img>0));
    mean_img = sum_img / count_img;
    var_img=var(img(img>0));
    sig_img = sqrt(var_img);
    
    if sig_img<0.1/255
        sig_img=0.1/255;
    end
    
    numSDevs = 3.0;
    %Rectify polarity=true
    meanGrey=0;
    range= numSDevs * sig_img;
    halfrange=0;
    rangenew = 255;
    %Rectify polarity=false
    %meanGrey=127 / 255;
    %range= 2*numSDevs * sig_img;
    %halfrange = numSDevs * sig_img;
    
    for i=1:m
        for j=1:n
            l=img(i,j);
            if l==0
                img(i,j)=meanGrey;
            end
            if l~=0
                f=(l+halfrange)*rangenew/range;
                if f>rangenew
                    f=rangenew;
                end
                if f<0
                    f=0;
                end
                img(i,j)= floor(f);
            end
        end
     end
    normalizedMat=img;
end