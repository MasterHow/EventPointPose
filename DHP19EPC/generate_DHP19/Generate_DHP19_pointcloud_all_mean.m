%--------------------------------------------------------------------------
% This script is modified from the constant count frame generation version of DHP19. 
% Use this script to generate the **Event Point Cloud data** of DHP19 dataset.
% ***MeanLabel Version***
% The script loops over all the DVS recordings and generates .h5 files
% of constant count frames.
%
% To import the aedat files here we use a modified version of 
% ImportAedatDataVersion1or2, to account for the camera index originating 
% each event.
%--------------------------------------------------------------------------

% Set the paths of code repository folder, data folder and output folder 
% where to generate files of accumulated events.
rootCodeFolder = 'F:\EventPointPose\DHP19EPC'; % root directory of the git repo.
rootDataFolder = 'F:\DHP19\'; % root directory of the data downloaded from resiliosync.
outDatasetFolder = 'F:\DHP19EPC_dataset\train_MeanLabel\'; % root directory of generated train data or test data.

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

global coordx_list_0
global coordy_list_0
global timeStamp_list_0
global pol_list_0

global coordx_list_1
global coordy_list_1
global timeStamp_list_1
global pol_list_1

global coordx_list_2
global coordy_list_2
global timeStamp_list_2
global pol_list_2

global coordx_list_3
global coordy_list_3
global timeStamp_list_3
global pol_list_3

global counter_cam0
global counter_cam1
global counter_cam2
global counter_cam3

nbcam_use = [0, 1, 2, 3];
eventsPerFullFrame = eventsPerFrame * 4;

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

            InitParameters();

            is_MeanLabel = true;     % true for MeanLabel��
                                     % false for LastLabel��
            
            % Initialization
            X_p = (X-cam*sx);
            y_p = y;

            count_all = 0;
            nbFrame_initialization = round(length(timeStamp)/eventsPerFullFrame);
            PointCloudMovie = NaN(4, eventsPerFullFrame, nbFrame_initialization); % Events for all cameras
            CamPointNumMovie = NaN(length(nbcam_use), nbFrame_initialization); % Event Point Numbers for each frame of each camera
            poseMovie = NaN(13, 3, nbFrame_initialization); % Label

            frame_num = 0;
            last_k = 1;    
            
            for time_idx = 1:length(timeStamp) 
                if cam(time_idx) == nbcam_use(1)
                    counter_cam0 = counter_cam0 + 1;
                    count_all = count_all + 1;
                    coordx_list_0(counter_cam0) = X_p(time_idx);
                    coordy_list_0(counter_cam0) = y(time_idx);
                    timeStamp_list_0(counter_cam0) = timeStamp(time_idx);
                    pol_list_0(counter_cam0) = pol(time_idx);

                end
                if cam(time_idx) == nbcam_use(2)
                    counter_cam1 = counter_cam1 + 1;
                    count_all = count_all + 1;
                    coordx_list_1(counter_cam1) = X_p(time_idx);
                    coordy_list_1(counter_cam1) = y(time_idx);
                    timeStamp_list_1(counter_cam1) = timeStamp(time_idx);
                    pol_list_1(counter_cam1) = pol(time_idx);

                end
                if cam(time_idx) == nbcam_use(3)
                    counter_cam2 = counter_cam2 + 1;
                    count_all = count_all + 1;
                    coordx_list_2(counter_cam2) = X_p(time_idx);
                    coordy_list_2(counter_cam2) = y(time_idx);
                    timeStamp_list_2(counter_cam2) = timeStamp(time_idx);
                    pol_list_2(counter_cam2) = pol(time_idx);

                end
                if cam(time_idx) == nbcam_use(4)
                    counter_cam3 = counter_cam3 + 1;
                    count_all = count_all + 1;
                    coordx_list_3(counter_cam3) = X_p(time_idx);
                    coordy_list_3(counter_cam3) = y(time_idx);
                    timeStamp_list_3(counter_cam3) = timeStamp(time_idx);
                    pol_list_3(counter_cam3) = pol(time_idx);

                end
                if count_all == eventsPerFullFrame
                    frame_num = frame_num + 1;

                    k = floor((timeStamp(time_idx) - startTime)*0.0001)+1; 
                    if k > length(XYZPOS.XYZPOS.head)
                        break;
                    end

                    cam0_frame = cat(1, coordx_list_0, sy-coordy_list_0, timeStamp_list_0, pol_list_0);
                    cam1_frame = cat(1, coordx_list_1, sy-coordy_list_1, timeStamp_list_1, pol_list_1);
                    cam2_frame = cat(1, coordx_list_2, sy-coordy_list_2, timeStamp_list_2, pol_list_2);
                    cam3_frame = cat(1, coordx_list_3, sy-coordy_list_3, timeStamp_list_3, pol_list_3);

                    frame_all = [cam0_frame, cam1_frame, cam2_frame, cam3_frame];
                    cam_points_num_all = [counter_cam0; counter_cam1; counter_cam2; counter_cam3];

                    PointCloudMovie(:,:,frame_num) = frame_all;
                    CamPointNumMovie(:,frame_num) = cam_points_num_all;
                    pose = GetPoseLabels(XYZPOS, last_k, k, is_MeanLabel);
                    poseMovie(:,:,frame_num) = pose;  

                    last_k = k;
                    count_all = 0;

                    InitParameters();
                    disp(strcat('Successfully extract No.', num2str(frame_num), 'frames'));
                end

            end
            
            % save h5 file
            data_fileName = strcat(output_data_Folder, '/', outDVSfile_subname);
            label_fileName = strcat(output_label_Folder, '/', outDVSfile_subname);
            if saveHDF5 == 1        
                DVSfilenameh5 = strcat(data_fileName,'.h5');
                PointCloudMovie = PointCloudMovie(:,:,1:frame_num);
                CamPointNumMovie = CamPointNumMovie(:,1:frame_num);
                if convert_labels == true
                    Labelsfilenameh5 = strcat(label_fileName,'_label.h5');
                    poseMovie = poseMovie(:,:,1:frame_num);
                end

                h5create(DVSfilenameh5,'/DVS',[4 eventsPerFullFrame frame_num]);
                h5write(DVSfilenameh5, '/DVS', PointCloudMovie); 
                h5create(DVSfilenameh5,'/CamPointNum',[length(nbcam_use) frame_num]);
                h5write(DVSfilenameh5, '/CamPointNum', CamPointNumMovie); 
                if convert_labels == true
                    h5create(Labelsfilenameh5,'/XYZ',[13 3 frame_num])
                    h5write(Labelsfilenameh5,'/XYZ',poseMovie)
                end
            end
            disp(strcat('Successfully save ', num2str(frame_num), ' frames and labels'));
            disp(strcat('******Finished****** ', outDVSfile_subname, '.aedat'));
            
        end % loop over movements
    end % loop over sessions
end % loop over subjects

disp(strcat('******Finished All Data****** '));

