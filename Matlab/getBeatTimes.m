function beatTimes = getBeatTimes(fileName, conf)
% Reads beat times from .beats.txt files previously analysed.

beatsDir = [conf.rootDir filesep 'Data' filesep 'salami-data-public-master' filesep 'beats'];
[~, fn, ~] = fileparts(fileName);
beatTimes = textread([beatsDir filesep fn '.beats.txt']);