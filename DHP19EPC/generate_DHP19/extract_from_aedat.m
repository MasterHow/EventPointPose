function [startIndex, stopIndex, ...
          pol_tmp3, X_tmp3, y_tmp3, cam_tmp3, timeStamp_tmp3] = ...
   extract_from_aedat(...
                aedat, events, ...
                startTime, stopTime, sx, sy, nbcam, ...
                thrEventHotPixel, dt, ...
                xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1, ...
                xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2)
    
    
    if events(end) > events(1)
        % startIndex: event right after startTime
        % stopIndex:  event right before stopTime
        startIndexes = find(events > startTime);
        stopIndexes = find(events < stopTime);
        startIndex = startIndexes(1);
        stopIndex  = stopIndexes(end);
    else
        %%% special case if timeStamp overflows(for S14_1_1) %%%
        startIndexes = find(events == startTime);
        stopIndexes = find(events == stopTime);
        startIndex = startIndexes(1);
        stopIndex  = stopIndexes(1);
    end
    
    pol       = aedat.data.polarity.polarity(startIndex:stopIndex);
    x         = aedat.data.polarity.x(startIndex:stopIndex);
    y_raw     = aedat.data.polarity.y(startIndex:stopIndex);
    cam       = aedat.data.polarity.cam(startIndex:stopIndex);
    timeStamp = uint32(events(startIndex:stopIndex));
    
    % remove events out of the boundaries.
    cond = (x<0) | (x>sx-1) | (y_raw<0) | (y_raw>sy-1) | (cam<0) | (cam>nbcam-1);
    x(cond)=[];
    y_raw(cond)=[];
    pol(cond)=[];
    cam(cond)=[];
    timeStamp(cond)=[];
    
    X = (sx-x)+cam*sx;
    y = sy-y_raw;
    
    % apply filters to the events.
    [X_tmp, y_tmp, timeStamp_tmp, pol_tmp, cam_tmp] = ...
        HotPixelFilter(X, y, timeStamp, pol, cam, sx*nbcam, sy, thrEventHotPixel);
    
    [X_tmp1, y_tmp1, timeStamp_tmp1, pol_tmp1, cam_tmp1] = ...
        BackgroundFilter(X_tmp, y_tmp, timeStamp_tmp, pol_tmp, cam_tmp, sx*nbcam, sy, dt);
    
    [X_tmp2, y_tmp2, timeStamp_tmp2, pol_tmp2, cam_tmp2] = ...
        maskRegionFilter(X_tmp1, y_tmp1, timeStamp_tmp1, pol_tmp1, cam_tmp1, ...
                         xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1);
    
    [X_tmp3, y_tmp3, timeStamp_tmp3, pol_tmp3, cam_tmp3] = ...
        maskRegionFilter(X_tmp2, y_tmp2, timeStamp_tmp2, pol_tmp2, cam_tmp2, ...
                         xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2);

end