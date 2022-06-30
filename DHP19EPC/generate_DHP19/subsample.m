function [Isub]=subsample(I,sizex,sizey,new_size_x,new_size_y, where_to_cut)
    block_w=floor(sizex/new_size_x);
    block_h=floor(sizey/new_size_y);
    %%% this cuts always initial part
    if strcmp(where_to_cut, 'begin')
        I_cut=I((sizex-new_size_x*block_w)+1:end,(sizey-new_size_y*block_h)+1:end);
    end
    %%% this cuts always final part
    if strcmp(where_to_cut, 'end')
        I_cut=I((1:(end-(sizex-new_size_x*block_w))),1:(end-(sizey-new_size_y*block_h)));
    end
    %%% better to cut the central part. Cut central part for cameras 0,2,3.
    %%% camera 1 has different subsample function
    if strcmp(where_to_cut, 'center')
        I_cut=I( ...
            ( (sizex-new_size_x*block_w)/2 + 1 : (end -(sizex-new_size_x*block_w)/2 ) ), ...
              (sizey-new_size_y*block_h)/2 + 1 : (end -(sizey-new_size_y*block_h)/2 ) );
    end
    
    Isub=zeros(new_size_x,new_size_y);
    for w=0:(new_size_x-1)
        for h=0:(new_size_y-1)
            x_start=w*block_w+1;
            y_start=h*block_h+1;
            Isub(w+1,h+1)=mean(mean(I_cut(x_start:x_start+block_w-1,y_start:y_start+block_h-1)));
        end
    end
end