% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [] = saveFile(matrix, name)
    fileID = fopen(sprintf('%s.bin',name), 'w');
    fwrite(fileID, matrix, 'double');
    fclose(fileID);
end

