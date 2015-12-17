addpath(genpath('Piotr'));

load train/train.mat;

i = randi(length(train.y));
figure
% load img
img = imread(sprintf('train/imgs/train%05d.jpg', i));

subplot(121);
imshow(img); % image itself

feature = hog( single(img)/255, 17, 8);
im( hogDraw(feature) ); colormap gray;
axis off; colorbar off;

