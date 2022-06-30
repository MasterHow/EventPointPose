function [x,y,t,pol,cam] = HotPixelFilter(x,y,t,pol,cam,xdim,ydim,threventhotpixel)
    %ignore the Hot Pixel
    %hot pixels are define as the pixels that record a number of event
    %bigger than threventhotpixel
    if nargin<7
        threventhotpixel= 100; %default value for timehotpixel us
    end
    
    hotpixelarray=zeros(xdim,ydim);
    
    for i=1:length(t)
        hotpixelarray(x(i),y(i))=hotpixelarray(x(i),y(i))+1;
    end
    
    selindexarray = hotpixelarray>= threventhotpixel;
    [hpx,hpy]=find(selindexarray);

    for k=1:length(hpx)
        selindexvector= x==hpx(k) & y==hpy(k);
        x=x(~selindexvector);
        y=y(~selindexvector);
        t=t(~selindexvector);
        pol=pol(~selindexvector);
        cam=cam(~selindexvector);
    end
    
end
