function Gallery = crop_all_faces(img_dir,img_save_dir,fileSaveName,crop_annotation_path)
fileLocation = fullfile(img_save_dir,strcat(fileSaveName,'.mat'));
if exist(fileLocation)
    load(fileLocation)
    return
end
fid = fopen(crop_annotation_path);
tline = fgetl(fid);
annotations = {};
i = 1;
padding = 60;
Gallery = containers.Map;
if ~exist(img_save_dir,'dir')
    mkdir(img_save_dir)
end
    
wb = waitbar(0,'Loading and Cropping Images');
while ischar(tline)
    annotations{i} = tline;
    tline = fgetl(fid);
    i = i+1;
end

for i = 1:length(annotations)
    
    tline = strtrim(annotations{i});
    try
        components = strsplit(tline);
    if length(components) >= 5
        fileName = components{1};
        x= components{2};
        y= components{3};
        w= components{4};
        h= components{5};
        r = [abs(str2num(x))-padding abs(str2num(y))-padding abs(str2num(w))+2*padding abs(str2num(h))+2*padding];
        imgFile = fullfile(img_dir,fileName);
        img = imread(imgFile);
        img = imcrop(img,r);
        if length(img) > 0
        Gallery(fileName) = img;
        imwrite(img,fullfile(img_save_dir,fileName));
        end
    end
    end
    waitbar(i/length(annotations),wb);

end
save(fileLocation,'Gallery');
fclose(fid);


