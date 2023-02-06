%% This script reads in the OPAL data and calculates spatiotemporal and upperbody parameters. 
% Make sure you run the code in order and if you make changes
% in a section, run the full script again. When you first run the script,
% it is best to run by section, so you can check the data for each step.

clearvars
clc
close all
cd('../raw_data/')    %Set directory to file you want 
addpath('../functions')                                  %if you have functions saved in a seperate folder within you matlab folder


%% Import files into a data structure
% Define where your data is and make sure the right functions are used to
% read you data in

cd('.');
MatlabFolder = pwd;

% filepath for data folder
DataFolder = '../raw_data/'; % Define folder where the data is. Data is saved in such a way that every participants has its own folder
% change directory to data folder
cd(DataFolder)
% folders = all folders within data folder
folders = dir('*');     % get direction participant folders

%Subj name
tmp={folders.name}';

% depending on what sort of files are in the folder, might need to change first number
%subjName=tmp(4:23); % subjName = ppt_id in project insole
subjName=tmp(4); % subjName = ppt_id in project insole <- INDEX 4 = PPT_001
clear('tmp')

%Select visit: CLINIC_v1, CLINIC_v5 OR v6 depending on your study need to
%change this
selectVisit='Baseline_week0'; %%% DO NOT NEED FOR MY STUFF
%selectVisit2='CLINIC_v6';

%Select test: TUG, pref, fast, Depending on your study, change this
selectTest='trial19'; %%% DO NOT NEED FOR MY STUFF

% loop through participants
for idsubj=1:length(subjName)
    disp(subjName{idsubj})
    % open the folder containing OPAL data
    tmp0=dir([subjName{idsubj}]); %Open OPAL folder for participant
    disp('tmp0')
    disp(tmp0)
    
    %extra check to see whether folder has OPAL data, if not will
    %skip participant
    if isempty(tmp0)
        NoOpal=subjName{idsubj}
        continue
    else
        
    %subjVisit={tmp0(9:end).name}'; % subjVisit = trial_id in project insole 
    subjVisit={tmp0(13).name}'; % DO ONE AT A TIME <- INDEX 11 = TRIAL07 <- trial number + 4
    disp(subjVisit)
    for idvisit=1:length(subjVisit)
        disp('subjvisit')
        disp(subjVisit{idvisit})
        % check that the subVisit and selectVisit are the same           
        %if strcmp(subjVisit{idvisit},selectVisit) %||strcmp(subjVisit{idvisit},selectVisit2)
        %if strcmp(subjVisit{idvisit},'trial05') %% REMOVED THIS IF
        %STATEMENTS AND REMOVED INDENT
        %%||strcmp(subjVisit{idvisit},selectVisit2) 
            % select the .h IMU data file
        cd([subjName{idsubj},'/',subjVisit{idvisit}]); %set file to data file
        %tmp = dir(['*',selectTest,'.h5']); %LVG Get the right test;
        tmp = dir(['*','.h5']); %LVG Get the right test;
        subjTest={tmp.name}';
        disp(subjTest)
        
        %LVG check whether test exist in the folder
        if isempty(subjTest)
            NoFile=subjName{idsubj}
            continue
        else
            subjTestFolder={tmp.folder}';
            filepath=[subjTestFolder{1} '/' subjTest{1}];  %LVG changed for UP study
            %data(idsubj)=import_h5_file(filepath,subjName{idsubj},selectVisit,selectTest);
            data(idsubj)=import_h5_file(filepath,subjName{idsubj},subjVisit{idvisit},subjVisit{idvisit});
            break
            if isempty(data(idsubj).Datasets(1).Accel)
                NoData=subjName(idsubj)
                data(idsubj)=[];
            end
        end 
        %else
        %    continue;                    
        %end
    %clear ('tmp0','tmp','subjVisit','subjTest','subjTestFolder');
    cd(DataFolder);
    end
    end
    subjinfold={data.SubjId}';
    disp('subjinfold:')
    disp(subjinfold)
    for n=1:length(subjinfold)
        if strfind(subjName{idsubj},data(n).SubjId)    
            Check(n)=1;
            disp('check(n) = 1')
        else
            Check(n)=0;
            disp('check(n) = 0')
        end
    end
    if sum(Check)==0
        NoOpal=subjName{idsubj}
    end
    clear ('tmp0','tmp','subjVisit','subjTest','subjTestFolder');
    cd(DataFolder);
end

%Remove empty folders
emptyIndex=find(arrayfun(@(data) isempty(data.SubjId),data));
data(emptyIndex)=[];
subjName(emptyIndex)=[];

SubjList=intersect(subjName,{data.SubjId});
OutSubjList=setxor(subjName,{data.SubjId});

cd(MatlabFolder);


%% 1) Settings
% Change these according to your study

%Upstudy=1;

Fs=data(1).Datasets.SampleRate;
Fs=double(Fs); %sample rate
T=1/Fs;
n=4;  %order Filter low pass


%Settings for removing motionless signals, change xi and xp if you want to
%change the window at the beginning and the end of which data is being
%removed
%sec=360;                  %trial duration (=6min)
%sec=342.6;
gyro=data(1).Datasets.Gyro
sec = length(gyro)/100  %trial duration (=6min)
sample_wmot=0.5;          %perc of samples without motion (=50%)
%xi=0.2;                   %sgn percentage to cut at the beging
xi=0;                   %sgn percentage to cut at the beging
%xp=0.2;                   %sgn percentage to cut at the end.
xp=1.0;                   %sgn percentage to cut at the end.
win=1*Fs;                 %1-sec window

%Settings for segmentation
setseg=1;                 %set 0 if you don't want segmentation
%L_corridor=8;         %corridor length [m]
L_corridor=6;         %corridor length [m] % length of walk


speed=1.6;                %mean normal speed [m/s] (pwMS=1.3; Ctrl=1.6)
h_peak=1;                 %peak_height [rad/s], a trunk rotation around the vertical plane with a minimum of 40�/sec (=0.7 rad/s)
turnThres=45;             %Turn angle higher than 45deg
turnThres180=115;         %Complete turn angle higher than 115deg


%% 2) Plot raw signals
cd('/Users/lukecleland/Documents/PhD/Research projects/Project insole/Data analysis/IMU data analysis');
%nameFolder='/Users/lukecleland/Documents/PhD/Research projects/Project insole/Data analysis/IMU data analysis/Raw data';
nameFolder=fullfile('/Users/lukecleland/Documents/PhD/Research projects/Project insole/Data analysis/IMU data analysis/Raw data', data.SubjId, selectTest);
mkdir(nameFolder)
cd(nameFolder)
plotRawSignals(data,[nameFolder])

csvwrite('left_foot_accel.csv', data.Datasets(3).Accel)


%% 5) Identify portions of motionless in the signal at the beginning and at the end of signal and remove them
% If to much or to little data is taken out check the settings (xi, xp)

bseg=[];                  %equals to [] if you want to find portions of motionless at the beginning and at the end of the signal
SubjList_wmot=[];
nameFolder='MotionLess plot';
mkdir(nameFolder)

for idsubj=1:size(data,2)
    Lsgn=length(data(idsubj).Datasets(1).Accel);
    [data,subj_wmot]=motionless(data,idsubj,bseg,win,Fs,nameFolder,sample_wmot,round(xi*Lsgn),round(xp*Lsgn),sec); %LVG added round since integer operands are required
    SubjList_wmot=[SubjList_wmot;subj_wmot];
    if isfield(data,'SgnMotionLess')
        if ~isempty(data(idsubj).SgnMotionLess)
            L=size(data(idsubj).SgnMotionLess,2);
            istart_all=[data(idsubj).SgnMotionLess(:).s_start];
            istop_all=[data(idsubj).SgnMotionLess(:).s_end];
            
            %motionless [1; xi]
            istartNow0=istart_all(istart_all<=xi*Lsgn);
            istopNow0=istop_all(istop_all<=xi*Lsgn);
            if ~isempty(istartNow0) && ~isempty(istopNow0)
                if istartNow0(1)~=1
                    istart0=1;
                else
                    istart0=istartNow0(1);
                end
                istop0=motionless_check(istartNow0,istopNow0,win);
                
                data(idsubj).SgnMotionLess(1).s_startUpd=istart0;
                data(idsubj).SgnMotionLess(1).s_endUpd=istop0;
                for idsens=1:size(data(idsubj).Datasets,2)
                    data(idsubj).Datasets(idsens).Accel(istart0:istop0,:)=NaN;
                    data(idsubj).Datasets(idsens).Gyro(istart0:istop0,:)=NaN;
                    data(idsubj).Datasets(idsens).Mag(istart0:istop0,:)=NaN;
                    data(idsubj).Datasets(idsens).Orient(istart0:istop0,:)=NaN;
                end
            end
            clear('istart0','istop0','istartNow0','istopNow0')
            
            %motionless [length(sgn)-xf; length(sgn)]
            if Lsgn>(sec*Fs)+1
               K=Lsgn-((Lsgn-(sec*Fs))+(xp*Lsgn));
            else
               K=Lsgn-(xp*Lsgn);
            end
            istartNowf=istart_all(istart_all>=K);
            istopNowf=istop_all(istop_all>=K);
            if ~isempty(istartNowf) && ~isempty(istopNowf)
                istartf=istartNowf(1);
                if ~isempty(find(istartNowf>(sec*Fs)+1,1)) || ~isempty(find(istopNowf>(sec*Fs)+1,1))
                    istopf=Lsgn;
                else
                    istopf=motionless_check(istartNowf,istopNowf,win);
                end
                
                if (Lsgn-istopf)<2*win+1
                    istopf=Lsgn;
                end
                
                data(idsubj).SgnMotionLess(2).s_startUpd=istartf;
                data(idsubj).SgnMotionLess(2).s_endUpd=istopf;
                for idsens=1:size(data(idsubj).Datasets,2)
                    data(idsubj).Datasets(idsens).Accel(istartf:istopf,:)=NaN;
                    data(idsubj).Datasets(idsens).Gyro(istartf:istopf,:)=NaN;
                    data(idsubj).Datasets(idsens).Mag(istartf:istopf,:)=NaN;
                    data(idsubj).Datasets(idsens).Orient(istartf:istopf,:)=NaN;
                end
            end
            clear('istartf','istopf','istartNowf','istopNowf')
        else
            continue;
        end
    end
end
clear('subj_wmot');

for idsubj=1:size(data,2)
    for idsens=1:size(data(idsubj).Datasets,2)
        data(idsubj).Datasets(idsens).Accel(any(isnan(data(idsubj).Datasets(idsens).Accel(:,1:3)),2),:)=[];
        data(idsubj).Datasets(idsens).Gyro(any(isnan(data(idsubj).Datasets(idsens).Gyro(:,1:3)),2),:)=[];
        data(idsubj).Datasets(idsens).Mag(any(isnan(data(idsubj).Datasets(idsens).Mag(:,1:3)),2),:)=[];
        data(idsubj).Datasets(idsens).Orient(any(isnan(data(idsubj).Datasets(idsens).Orient(:,1:3)),2),:)=[];        
    end
end




%% 6) Check length signals and gaps

L_sgn=[];
L_h6min=[];
SubjList_h6min=[];
for idsubj=1:size(data,2)
    idsens=1;
    subjName={data(idsubj).SubjId};
    L_sgn(idsubj,1)=length(data(idsubj).Datasets(idsens).Accel);
    if L_sgn(idsubj)>sec*Fs+1
        SubjList_h6min=[SubjList_h6min;subjName];
        L_h6min=[L_h6min;L_sgn(idsubj)];
        for idsens=1:size(data(idsubj).Datasets,2)
            data(idsubj).Datasets(idsens).Accel(sec*Fs+1:end,:)=[];
            data(idsubj).Datasets(idsens).Gyro(sec*Fs+1:end,:)=[];
            data(idsubj).Datasets(idsens).Mag(sec*Fs+1:end,:)=[];
            data(idsubj).Datasets(idsens).Orient(sec*Fs+1:end,:)=[];
        end
    end
end

save([selectTest '_' selectVisit '_input'],'data');



%% 7) Align IMUs & check directions
%Check wether the IMU/accelerometer was placed the right way around

SensorLabel=[];
PossibleAnkleWrong_all=[];
AnkleWrong_all=[];
PossibleUBWrong_all=[];
UB_V_Wrong_all=[];

for idsubj=1:size(data,2)
    subjName=data(idsubj).SubjId;
    for idsens=1:size(data(idsubj).Datasets,2)
        SensorLabel={data(idsubj).Datasets(idsens).Label};
        if contains(SensorLabel,'Ankle')
            Fc=8;  %cut off frequency
            [accelC,gyroC,AnkleWrong,PossibleAnkleWrong] = ...
                align_ankleIMU(Fc,Fs,n,SensorLabel,data(idsubj).Datasets(idsens).Accel,data(idsubj).Datasets(idsens).Gyro,subjName);
            PossibleAnkleWrong_all=[PossibleAnkleWrong_all;PossibleAnkleWrong];
            AnkleWrong_all=[AnkleWrong_all;AnkleWrong];
            data(idsubj).Datasets(idsens).Accel=accelC;
            data(idsubj).Datasets(idsens).Gyro=gyroC;
            clear('PossibleAnkleWrong','AnkleWrong');
        elseif  strcmp(SensorLabel,'Chest') || strcmp(SensorLabel,'Forehead') || strcmp(SensorLabel,'Neck') || strcmp(SensorLabel,'LowerBack') 
            Fc=1.5;  %cut off frequency
            [data(idsubj).Datasets(idsens).Accel,data(idsubj).Datasets(idsens).Gyro,UBWrongV]= ...
                align_upperbodyIMU_V(SensorLabel,data(idsubj).Datasets(idsens).Accel,data(idsubj).Datasets(idsens).Gyro,...
                subjName);
            UB_V_Wrong_all=[UB_V_Wrong_all;UBWrongV];
            clear('UBWrongV');
        end
    end
end
close all;

save([selectTest '_' selectVisit '_QC'],'data');



%% 8) Segmentation 
% segment data automatically segments by turns
% This wil split the data up in walking bouts based on the gyro signal. In
% the first subplot you will see the filtered angular velocity. The red
% stars should be placed around the peaks. In the second subplot you see
% the yaw angle and the red lines should be placed around the
% incline/decline in the signal. If this is not the case, press no and a
% graph will come up. You will be asked to click three times in this graph.
% The first two clicks are to define the likely length of the walking bout,
% so select for example a point at the beginning and then close to the
% first peak. The third click is to define the minimum higth of the peak.

if setseg==1
    Fc=1.5;  %cut off frequency
    nameFolder='Segmentation plot';
    mkdir(nameFolder)
    for idsubj=1:size(data,2)
        [data]=turning_detection(data,nameFolder,idsubj,Fc,Fs,h_peak,L_corridor,speed,turnThres,turnThres180);
    end
    save([selectTest '_' selectVisit '_seg'],'data');
else
    for idsubj=1:size(data,2)
        data(idsubj).Passes=struct('passID',1,'s_start',1,'s_end',length(data(idsubj).Datasets(1).Gyro));
        assignin('base','data',data) %Forces the updated variable into workspace (make global)
    end
end


%% 9) Identify portions of motionless in the signal

bseg=1;                   %equals to [] if you want to find portions of motionless at the beginning and at the end of the signal
xc=1;                     %number of samples to cut signal at the beginning and at the end.
win=Fs; 

SubjList_wmot=[];
nameFolder='MotionLess plot_int';
mkdir(nameFolder)
for idsubj=1:size(data,2)
    [data,subj_wmot]=motionless(data,idsubj,bseg,win,Fs,nameFolder,sample_wmot,xc,sec);
    SubjList_wmot=[SubjList_wmot;subj_wmot];
    if isfield(data,'SgnMotionLessInt')
        if ~isempty(data(idsubj).SgnMotionLessInt)
            L=size(data(idsubj).SgnMotionLessInt,2);
            for j=1:L
                istart=data(idsubj).SgnMotionLessInt(j).s_start;
                istop=data(idsubj).SgnMotionLessInt(j).s_end;
                if istart>(sec*Fs)+1 || istop>(sec*Fs)+1
                    istop=length(data(idsubj).Datasets(1).Accel);
                    data(idsubj).SgnMotionLessInt(j).s_start=istart;
                    data(idsubj).SgnMotionLessInt(j).s_end=istop;
                    ind=j+1;
                    if ind<=L
                    data(idsubj).SgnMotionLessInt(ind:end)=[];
                    end
                    for idsens=1:size(data(idsubj).Datasets,2)
                        data(idsubj).Datasets(idsens).Accel(istart:istop,:)=NaN;
                        data(idsubj).Datasets(idsens).Gyro(istart:istop,:)=NaN;
                        data(idsubj).Datasets(idsens).Mag(istart:istop,:)=NaN;
                        data(idsubj).Datasets(idsens).Orient(istart:istop,:)=NaN;
                    end
                    break;
                else
                    for idsens=1:size(data(idsubj).Datasets,2)
                        data(idsubj).Datasets(idsens).Accel(istart:istop,:)=NaN;
                        data(idsubj).Datasets(idsens).Gyro(istart:istop,:)=NaN;
                        data(idsubj).Datasets(idsens).Mag(istart:istop,:)=NaN;
                        data(idsubj).Datasets(idsens).Orient(istart:istop,:)=NaN;
                    end
                end
            end
        end
    else
        continue;
    end
end
clear('subj_wmot');

