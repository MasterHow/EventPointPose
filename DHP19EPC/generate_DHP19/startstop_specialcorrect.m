function [startTime, stopTime] = startstop_specialcorrect(aedat, save_log_special_events, events, XYZPOS)

%%% conditions on special events %%%
% special events
% correct the startTime and stopTime
try
    specialEvents = int64(aedat.data.special.timeStamp);
    numSpecialEvents = length(specialEvents);

    if save_log_special_events
        % put the specialEvents to string, to print to file.
        specials_='';
        for k = 1:numel(specialEvents)
            specials_ = [specials_ ' ' num2str(specialEvents(k))];
        end
        fprintf(fileID_specials, '%s \t %s\n', aedatPath, specials_); 
    end % log_special_events

    %%% Special events cases: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % a) 1 without special event field (S14/session5/mov3)
    %
    % b) 2 with just 1 special event:
    %    S5/session4/mov2: 
    %                     special     1075788910
    %                     min(events) 1055513693
    %                     max(events) 1076195668 
    %    S16/session2/mov4:
    %                     special     278928206
    %                     min(events) 258344886
    %                     max(events) 279444627
    %    in both cases, special is closer to the end of the
    %    recording, hence we assume the initial special is
    %    missing.
    %
    % c) 326 with 2 special events: these are all 2
    %    consecutive events (or two at the same timestamp).
    %
    % d) 225 with 3 special events: first two are
    %    consecutive, the 3rd is the stop special event.
    %
    % e) 2 with 4 special events: 
    %    first 3 are equal, 4th is 1 timestep after the first.
    %    Same as c)
    %    S4/session5/mov7: 
    %                     special     149892732(x3), 149892733
    %                     min(events) 146087513
    %                     max(events) 170713237
    %    S12/session5/mov4:
    %                     special     411324494(x3), 411324495
    %                     min(events) 408458645
    %                     max(events) 429666644
    %    in both cases the special events are closer to the
    %    start of the recording, hence we assume the final 
    %    special is missing.
    % 
    % f) 3 with 5 special events: first 3 (or 4) are equal, 
    %    1 (or 0) right after the first, the last event is the final
    %    Same as d)
    %    (S5/session1/mov4, S5/session2/mov4, S5/session3/mov3).
    %
    % g) 2 with >700 special events (S4/session3/mov4, S4/session3/mov6)
    %    -> these recordings are corrupted, removed from DHP19
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % use head joint to calculate total number of timesteps
    % when information is missing from special events.
    % 1e4 factor is to go from 100Hz Vicon sampling freq to 
    % us DVS temporal resolution.
    n = length(XYZPOS.XYZPOS.head)*10000;

    if numSpecialEvents == 0
        % the field aedat.data.special does not exist
        % for S14_5_3. There are no other cases.
        error('special field is there but is empty');

    elseif numSpecialEvents == 1

        if (specialEvents-min(events)) > (max(events)-specialEvents)
            % The only event is closer to the end of the recording.
            stopTime = specialEvents;
            startTime = floor(stopTime - n);
        else
            startTime = specialEvents;
            stopTime = floor(startTime + n);
        end


    elseif (numSpecialEvents == 2) || (numSpecialEvents == 4)
        % just get the minimum value, the others are max 1
        % timestep far from it.
        special = specialEvents(1); %min(specialEvents);

        %%% special case, for S14_1_1 %%%
        % if timeStamp overflows, then get events only
        % until the overflow.
        if events(end) < events(1)
            startTime = special;
            stopTime = max(events);


        %%% regular case %%%
        else
            if (special-events(1)) > (events(end)-special)
                % The only event is closer to the end of the recording.
                stopTime = special;
                startTime = floor(stopTime - n);
            else
                startTime = special;
                stopTime = floor(startTime + n);
            end
        end 

    elseif (numSpecialEvents == 3) || (numSpecialEvents == 5)
        % in this case we have at least 2 distant special
        % events that we consider as start and stop.
        startTime = specialEvents(1);
        stopTime = specialEvents(end);

    elseif numSpecialEvents > 5
        % Two recordings with large number of special events.
        % Corrupted recordings, skipped.
        % continue 
    end

catch 
    % if no special field exists, get first/last regular
    % events (not tested).
    startTime = events(1); 
    stopTime = events(end); 

    if save_log_special_events
        disp(strcat("** Field 'special' does not exist: ", aedatPath));
        fprintf(fileID_specials, '%s \t\n', aedatPath);
    end %
end % end try reading special events