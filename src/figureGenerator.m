frontalizers = {'Vito' 'Hassner'}
landmarkers = {'dlib' 'ZhuRamanan' 'CMR'}
folder = '/Users/joel/Documents/ICB-RW/data/icbrw_Data/icbrw_GalleryImages/'
images = dir(strcat(folder, '*.jpg'));
parfor i = 1:length(images)
for f = frontalizers
    for l = landmarkers
        name = images(i);
        name = name.name
       makeFrontalFigures(strcat(folder,name), l{1}, f{1})

    end
end
end