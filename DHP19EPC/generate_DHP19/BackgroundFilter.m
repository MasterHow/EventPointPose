function [x,y,t,pol,cam] =BackgroundFilter(x,y,t,pol,cam,xdim,ydim,dt)
    %filter out the events that are not support the neighborhood events
    %dt, define the time to consider an event valid or not
    %if nargin<7
    %    dt= 30000; %default value for dt us
    %end

    lastTimesMap=zeros(xdim,ydim);
    index=zeros(length(t),1);
    for i=1:length(t)
        ts=t(i); xs=x(i); ys=y(i);
        deltaT=ts-lastTimesMap(xs,ys);
        
        if deltaT>dt
            index(i)=NaN;
        end
        
        if ~(xs==1 || xs==xdim || ys==1 || ys==ydim)
            lastTimesMap(xs-1, ys) = ts;
            lastTimesMap(xs+1, ys) = ts;
            lastTimesMap(xs, ys-1) = ts;
            lastTimesMap(xs, ys+1) = ts;
            lastTimesMap(xs-1, ys-1) = ts;
            lastTimesMap(xs+1, ys+1) = ts;
            lastTimesMap(xs-1, ys+1) = ts;
            lastTimesMap(xs+1, ys-1) = ts;
        end
    end
    
    x(isnan(index))=[];
    y(isnan(index))=[];
    t(isnan(index))=[];
    pol(isnan(index))=[];
    cam(isnan(index))=[];
    
end
