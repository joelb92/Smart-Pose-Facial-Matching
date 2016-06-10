function run_ICBRW_experiment(probePath,annotationPath,watchlistPath,watchlistAnnotationPath)
%Load the waitlist
display('Loading and cropping watchlist faces...')
wlistGalleryDir = watchlistPath%'../data/icbrw_Data/icbrw_GalleryImages/';
wlistCroppedDir = fullfile(watchlistPath,'../icbrw_GalleryImages_cropped');
wlistAnnotationsFile = watchlistAnnotationPath%'../data/icbrw_Data/annotations_GalleryImages.txt';
wlistSaveFileName = 'GalleryFile';
padding = 60;
watchlistFaces = crop_all_faces(wlistGalleryDir,wlistCroppedDir,wlistSaveFileName,wlistAnnotationsFile);
display('DONE!')

landmarker = 'CMR'
hardsym_type = 1; %Set to 1, 2, or 3 for different hard-symmetry paddings
%build frontalized versions of each face

watchlistKeys = watchlistFaces.keys;
display('Frontalizing Watchlist...')
wb = waitbar(0,'Frontalizing Watchlist...');
%No symmetry (hassner): 1  No symmetry (vito):2  right hard symmetry (right side of face is mirrored): 3
%left hard symmetry (left side of face is mirrored): 4
addpath('Vito/');
rsize = 1;
frontalizedGalleryFileName = fullfile(wlistCroppedDir,strcat('Gallery_CMR_hsym',num2str(hardsym_type),'.mat'))
if exist(frontalizedGalleryFileName,'file')
    load(frontalizedGalleryFileName)
else
    removeKeys = {};
    for i = 1:length(watchlistKeys)
        wkey = watchlistKeys{i};
        ID = wkey(1:3);
        pose = wkey(5);
        if strcmp(pose,'f')
            resize()
            img = watchlistFaces(wkey);
            imgPath = fullfile(wlistCroppedDir,wkey);
            [fidu_XY,frontal_raw_hassner,frontal_sym,hardsym_images_hassner]=demo(img,'ZhuRamanan',imgPath,[]);
            try
            [warped_surface, landmarked_img,frontal_raw_vito,hardsym_images]=demoVito(img,'dlib',imgPath,[padding/2, padding/2,size(img,2)-padding,size(img,1)-padding]);
            catch
                warped_surface = [];
                landmarked_img = [];
                frontal_raw_vito = [];
                hardsym_images = [];
            end
            
            images = cell(1,4);
            fsuccess = 1;
            if isempty(frontal_raw_hassner);
                images{1} = img;
                
                fsuccess = 0;
            end
            if isempty(frontal_raw_vito);
                images{2} = img;
                images{3} = img;
                images{4} = img;
                fsuccess = 0;
            end
            if fsuccess
                images{1} = frontal_raw_hassner;
                images{2} = frontal_raw_vito;
                images{3} = hardsym_images{2*hardsym_type};
                images{4} = hardsym_images{1*hardsym_type};
            end
            if isempty(images{1})
                images{1} = img;
            end
            if isempty(images{2})
                images{2} = img;
            end
            if isempty(images{3})
                images{3} = img;
            end
            if isempty(images{4})
                images{4} = img;
            end
            
            watchlistFaces(wkey) = images;
        else
            removeKeys{rsize} = wkey;
        
        end
        waitbar(i/length(watchlistKeys),wb);
    end
    remove(watchlistFaces,removeKeys);
    save(frontalizedGalleryFileName,'watchlistFaces');
end
display('DONE!')
watchlistKeys = watchlistFaces.keys;
watchSaveDir = strcat(wlistCroppedDir,'_frontal');
mkdir(watchSaveDir);
display('Writing out all frontalized watchlist files...')
for i = 1:length(watchlistKeys)
    wkey = watchlistKeys{i};
     pose = wkey(5);
        if strcmp(pose,'f')
    images = watchlistFaces(wkey);
    imwrite(images{1},fullfile(watchSaveDir,strcat(wkey(1:5),'_1.jpg')));
    imwrite(images{2},fullfile(watchSaveDir,strcat(wkey(1:5),'_2.jpg')));
    imwrite(images{3},fullfile(watchSaveDir,strcat(wkey(1:5),'_3.jpg')));
    imwrite(images{4},fullfile(watchSaveDir,strcat(wkey(1:5),'_4.jpg')));
        end
end
display('DONE!')

display('Loading and cropping probe faces...')
plistDir = fullfile(probePath)% '../data/icbrw_Data/icbrw_ProbeImages/';
plistCroppedDir = fullfile(probePath,'../icbrw_ProbeImages_cropped');
plistAnnotationsFile = fullfile(annotationPath);
plistSaveFileName = 'ProbeFile';
probeFaces = crop_all_faces(plistDir,plistCroppedDir,plistSaveFileName,plistAnnotationsFile);
display('DONE!');

probeKeys = probeFaces.keys;
display('Frontalizing Probe List...')
wb = waitbar(0,'Frontalizing Probe Images...');
%No symmetry (hassner): 1  No symmetry (vito):2  right hard symmetry (right side of face is mirrored): 3
%left hard symmetry (left side of face is mirrored): 4
rsize = 1;
failures = 0;
frontalizedProbeSetFileName = fullfile(plistCroppedDir,strcat('Probe_CMR_hsym',num2str(hardsym_type),'.mat'));
if exist(frontalizedProbeSetFileName,'file')
    load(frontalizedProbeSetFileName);
else
    removeKeys = {};
    
    for i = 1:length(probeKeys)
        frontal_sym = [];
        wkey = probeKeys{i}
        ID = wkey(1:3);
        imNum = wkey(5:6);
        img = probeFaces(wkey);
        imgPath = fullfile(plistCroppedDir,wkey);
        matchWith = -1;
        hardsym_images = {};
        frontal_sym = [];
        if ~isempty(img)
            try
                [warped_surface, landmarked_img,frontal_raw,hardsym_images]=demoVito(img,'dlib',imgPath,[padding, padding,size(img,2)-padding*2,size(img,1)-padding*2]);
            catch
                display('could not frontalize image using vito! Trying Hassner instead...');
                try
                [fidu_XY,frontal_raw,frontal_sym,hardsym_images]=demo(img,'ZhuRamanan',imgPath,[]);
                catch
                   display('could not frontalize image using Hassner, either!')
                   matchWith = 2;
                   main_img = img;
                end
                
            end
            try
                [angle bbox fidu_XY] = estimatePoseZR(img);
            catch
                display('Could not estimate pose!');
                fidu_XY = [];
            end
            main_img = [];
            
            if length(hardsym_images) > 0 && length(fidu_XY) > 0
                if angle > 45
                    main_img = hardsym_images{1}
                    matchWith = 4;
                elseif angle <= 45 && angle > 15
                    main_img = uint8(frontal_sym);
                    if isempty(main_img)
                        try
                            [fidu_XY,frontal_raw,frontal_sym,hardsym_images]=demo(img,landmarker,imgPath,fidu_XY);
                            main_img = frontal_sym;
                        catch
                            display('could not frontalize image! angle 45 to 15')
                            main_img = img;
                            failures = failures+1
                        end
                        matchWith = 2;
                    end
                elseif angle <= 15 && angle >= -15
                    main_img = frontal_raw;
                    matchWith = 1;
                elseif angle >= -45 && angle < -15
                    main_img = uint8(frontal_sym);
                    if isempty(main_img)
                        try 
                            [fidu_XY,frontal_raw,frontal_sym,hardsym_images]=demo(img,landmarker,imgPath,fidu_XY);
                            main_img = frontal_sym;
                        catch
                            display('could not frontalize image! angle -45 to -15')
                            main_img = img;
                            failures = failures+1
                            
                        end
                        
                        matchWith = 2;
                    end
                elseif angle < -45
                    main_img = hardsym_images{2}
                    matchWith = 3;
                end
            else
                display('Could not frontalize, using original image instead')
                main_img = img;
                matchWith = 2;
                failures = failures+1
            end
            img_strct.matchWith = matchWith;
            img_strct.image = main_img;
            waitbar(i/length(probeKeys),wb);
            probeFaces(wkey) = img_strct;
        end
    end
end

probeSaveDir = strcat(plistCroppedDir,'_frontal');
mkdir(probeSaveDir);
fileID = fopen(fullfile(probeSaveDir,'posetypes.txt'),'w');
for i = 1:length(probeKeys)
    wkey = probeKeys{i};
    images = probeFaces(wkey);
    imwrite(images.image,fullfile(probeSaveDir,strcat(wkey(1:6),'_frontal.jpg')));

fprintf(fileID,'%i\n',images.matchWith);

end
fclose(fileID);

failures = failures
save(frontalizedProbeSetFileName,'probeFaces');
end


