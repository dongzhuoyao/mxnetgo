%src = '/data_a/dataset/AerialImageDataset/train/images/';
%src = '/data_a/dataset/AerialImageDataset/train/gt/';
src = '/data_a/dataset/AerialImageDataset/test/images/';


listing = dir([src,'*.tif']);
fileSum = length(listing);
for imgNum=1:fileSum
    disp(imgNum);disp(fileSum);
    img = imread(strcat(src,listing(imgNum).name));
    a = strrep(listing(imgNum).name,'.tiff','');
    a = strrep(listing(imgNum).name,'.tif','');
    imwrite(img,[src a '_tif2jpg.jpg']);
end