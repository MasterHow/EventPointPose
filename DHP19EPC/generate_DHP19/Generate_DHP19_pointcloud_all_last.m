%--------------------------------------------------------------------------
% This script is modified from the constant count frame generation version of DHP19. 
% Use this script to generate the **Event Point Cloud data** of DHP19 dataset.
% ***LastLabel Version***
% The script loops over all the DVS recordings and generates .h5 files
% of constant count frames.
%
% To import the aedat files here we use a modified version of 
% ImportAedatDataVersion1or2, to account for the camera index originating 
% each event.
%--------------------------------------------------------------------------

% Set the paths of code repository folder, data folder and output folder 
% where to generate files of accumulated events.
rootCodeFolder = 'F:\EventPointPose\DHP19EPC'; % root directory of the git repo to this folder.
rootDataFolder = 'F:\DHP19\'; % root directory of the data downloaded from resiliosync.
outDatasetFolder = 'F:\DHP19EPC_dataset\train_LastLabel\'; % root directory of generated train data or test data.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cameras number and resolution. Constant for DHP19.
nbcam = 4;
sx = 346;
sy = 260;

%%%%%%%%%%% PARAMETERS: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Average num of events per camera, for constant count frames.
eventsPerFrame = 7500; 

% Flag and sizes for subsampling the original DVS resolution.
% If no subsample, keep (sx,sy) original img size.
do_subsampling = false;
reshapex = sx;
reshapey = sy;

% Flag to save accumulated recordings.
saveHDF5 = true;

% Flag to convert labels
convert_labels = true;

save_log_special_events = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Hot pixels threshold (pixels spiking above threshold are filtered out).
thrEventHotPixel = 1*10^4;

% Background filter: events with less than dt (us) to neighbors pass through.
dt = 70000;

%%% Masks for IR light in the DVS frames.
% Mask 1
xmin_mask1 = 780;
xmax_mask1 = 810;
ymin_mask1 = 115;
ymax_mask1 = 145;
% Mask 2
xmin_mask2 = 346*3 + 214;
xmax_mask2 = 346*3 + 221;
ymin_mask2 = 136;
ymax_mask2 = 144;

%%% Paths     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t = datetime('now','Format','yyyy_MM_dd''_''HHmmss');
%
DVSrecFolder = fullfile(rootDataFolder,'DVS_movies/');
viconFolder = fullfile(rootDataFolder,'Vicon_data/');

% output directory where to save files.
out_data_folder_append = ['data'];
out_label_folder_append = ['label'];

addpath(fullfile(rootCodeFolder, 'read_aedat/'));
addpath(fullfile(rootCodeFolder, 'generate_DHP19/'));

% Setup output folder path.
output_data_Folder = fullfile(outDatasetFolder, out_data_folder_append);
output_label_Folder = fullfile(outDatasetFolder, out_label_folder_append);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numConvertedFiles = 0;

% setup output folder
if ~exist(output_data_Folder,'dir'), mkdir(output_data_Folder); end
% cd(output_data_Folder)

if ~exist(output_label_Folder,'dir'), mkdir(output_label_Folder); end
% cd(output_label_Folder)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop over the subjects/sessions/movements.

numSubjects = 12;
numSessions = 5;

fileIdx = 0;

% Loop for train data or test data

for subj = 1:numSubjects
% for subj = [13,14,15,16,17]
    subj_string = sprintf('S%d',subj);
    sessionsPath = fullfile(DVSrecFolder, subj_string);
    
    for sess = 1:numSessions
        sessString = sprintf('session%d',sess);

        movementsPath = fullfile(sessionsPath, sessString);
        
        if     sess == 1, numMovements = 8;
        elseif sess == 2, numMovements = 6;
        elseif sess == 3, numMovements = 6;
        elseif sess == 4, numMovements = 6;
        elseif sess == 5, numMovements = 7;
        end
        
        for mov = 1:numMovements
            fileIdx = fileIdx+1;
            
            movString = sprintf('mov%d',mov);

            aedatPath = fullfile(movementsPath, strcat(movString, '.aedat'));
            
            outDVSfile_subname = strcat(subj_string,'_',sessString,'_',movString);
            disp(strcat('******Start Extracting ', outDVSfile_subname, '.aedat******'));
            % skip iteration if recording is missing.
            if not(isfile(aedatPath)==1)
                continue
            end
            
            disp([num2str(fileIdx) ' ' aedatPath]);
            
            aedat = ImportAedat([movementsPath '/'], strcat(movString, '.aedat'));
            events = int64(aedat.data.polarity.timeStamp);
            
            labelPath = fullfile(viconFolder, strcat(subj_string,'_',num2str(sess),'_',num2str(mov), '.mat'));
            assert(isfile(labelPath)==1);
            XYZPOS = load(labelPath);
            
            
            
            [startTime, stopTime] = startstop_specialcorrect(aedat, save_log_special_events, events, XYZPOS);

            startTime = uint32(startTime);
            stopTime  = uint32(stopTime);

            [startIndex, stopIndex, pol, X, y, cam, timeStamp] = ...
                    extract_from_aedat(...
                                    aedat, events, ...
                                    startTime, stopTime, ...
                                    sx, sy, nbcam, ...
                                    thrEventHotPixel, dt, ...
                                    xmin_mask1, xmax_mask1, ymin_mask1, ymax_mask1, ...
                                    xmin_mask2, xmax_mask2, ymin_mask2, ymax_mask2);
            
            is_MeanLabel = false;       % true for MeanLabel；
                                        % false for LastLabel；

            % Initialization
            X_p = (X-cam*sx); 
            y_p = y;
            
            % eventsPerFrame 7500 for each camera frame seperately
            
            for cam_num = 0 : nbcam - 1

                cam_index = (cam == cam_num);
                X_p_cam = X_p(cam_index);
                y_cam = y_p(cam_index);
                timeStamp_cam = timeStamp(cam_index);
                pol_cam = pol(cam_index);
                nbFrame_initialization = round(length(timeStamp_cam)/eventsPerFrame);
                PointCloudMovie_cam = NaN(4, eventsPerFrame, nbFrame_initialization);

                poseMovie_cam = NaN(13, 3, nbFrame_initialization);

                for frame_num = 1 : nbFrame_initialization
                    temp_idx_diff = eventsPerFrame * (frame_num - 1);
                    coordx_list = [];
                    coordy_list = [];
                    timeStamp_list = [];
                    pol_list = [];
                    if (temp_idx_diff + 1 > length(timeStamp_cam)) || (temp_idx_diff + eventsPerFrame > length(timeStamp_cam))
                        frame_num = frame_num - 1;
                        break;
                    end
                    last_k = floor((timeStamp_cam(temp_idx_diff + 1) - startTime)*0.0001)+1;
                    k = floor((timeStamp_cam(temp_idx_diff + eventsPerFrame) - startTime)*0.0001)+1;
                    % if k is larger than the label at the end of frame
                    % accumulation, the generation of frames stops.
                    if k > length(XYZPOS.XYZPOS.head)
                        frame_num = frame_num - 1;
                        break;
                    end
                    for idx = 1 : eventsPerFrame

                        idxf = idx + temp_idx_diff;    
                        coordx_list(idxf - temp_idx_diff) = X_p_cam(idxf);
                        coordy_list(idxf - temp_idx_diff) = y_cam(idxf);
                        timeStamp_list(idxf - temp_idx_diff) = timeStamp_cam(idxf);
                        pol_list(idxf - temp_idx_diff) = pol_cam(idxf);

                    end

                    PointCloudMovie_cam(:,:,frame_num) = [coordx_list; timeStamp_list; sy-coordy_list; pol_list];         
                    pose = GetPoseLabels(XYZPOS, last_k, k, is_MeanLabel);
                    poseMovie_cam(:,:,frame_num) = pose;  
                    disp(strcat('Successfully extract No.', num2str(frame_num), 'frame for camera_No.', num2str(cam_num)));
                end
               

                % save h5 file
                data_fileName = strcat(output_data_Folder, '/', outDVSfile_subname, '_cam', num2str(cam_num));
                label_fileName = strcat(output_label_Folder, '/', outDVSfile_subname, '_cam', num2str(cam_num));
                if saveHDF5 == 1        
                    DVSfilenameh5 = strcat(data_fileName,'.h5');
                    PointCloudMovie_cam = PointCloudMovie_cam(:,:,1:frame_num);

                    if convert_labels == true
                        Labelsfilenameh5 = strcat(label_fileName,'_label.h5');
                        poseMovie_cam = poseMovie_cam(:,:,1:frame_num);
                    end

                    h5create(DVSfilenameh5,'/DVS',[4 eventsPerFrame frame_num]);
                    h5write(DVSfilenameh5, '/DVS', PointCloudMovie_cam); 
                    if convert_labels == true
                        h5create(Labelsfilenameh5,'/XYZ',[13 3 frame_num])
                        h5write(Labelsfilenameh5,'/XYZ',poseMovie_cam)
                    end
                end
                disp(strcat('Successfully save ', num2str(frame_num), ' frames and labels for camera_No.', num2str(cam_num)));
            end
            disp(strcat('******Finished****** ', outDVSfile_subname, '.aedat'));
            
        end % loop over movements
    end % loop over sessions
end % loop over subjects

disp(strcat('******Finished All Data****** '));

