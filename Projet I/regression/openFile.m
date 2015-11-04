% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function [matrix] = openFile(name, size)
    fileID = fopen(sprintf('%s.bin',name));
    matrix = fread(fileID, size, 'double');
    fclose(fileID);
end

