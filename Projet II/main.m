addpath(genpath('Piotr'));

load train/train.mat;

i = randi(length(train.y));
figure
% load img
img = imread(sprintf('train/imgs/train%05d.jpg', i));

subplot(121);
imshow(img); % image itself

subplot(122);
feature = reshape(train.X_hog(i,:), 13, 13, 32);
im( hogDraw(feature) ); colormap gray;
axis off; colorbar off;
