% Written by Diego Antognini & Jason Racine, EPFL 2015
% all rights reserved

function setSeed(seed)
    % set seed
	global RNDN_STATE  RND_STATE
	RNDN_STATE = randn('state');
	randn('state',seed);
	RND_STATE = rand('state');
	%rand('state',seed);
	rand('twister',seed);
end